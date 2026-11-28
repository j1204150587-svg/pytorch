"""Compare independent processes using the same patched wheel, with one descriptor switch."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def worker(args):
    import torch
    import torch_npu
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    library=Path(torch_npu.__file__).parent/"lib/libtorch_npu.so"
    dll=ctypes.CDLL(str(library))
    counter=dll.npu_addmm_rebase_counter
    counter.argtypes=[ctypes.c_uint]
    counter.restype=ctypes.c_uint64
    read=lambda: [counter(i) for i in range(4)]
    layout=json.loads(args.layout.read_bytes())
    sizes=json.loads(args.schedule.read_bytes())
    g=torch.Generator().manual_seed(1234)
    flats=[torch.randn(layout["flat"]["elements"],generator=g).bfloat16().npu() for _ in range(2)]
    def view(flat,geom):
        return flat.as_strided(geom["shape"],geom["stride"],geom["storage_element_offset"])
    banks=[[(view(flat,row["bias"]),view(flat,row["transposed_weight"])) for row in layout["projection_views"]] for flat in flats]
    xs=[{k:torch.randn(b*l,k,generator=g).bfloat16().npu() for k in (64,192)} for b,l in sizes]
    inputs=[{k:torch.randn(b*l,k,generator=g).bfloat16().npu() for k in (64,256)} for b,l in sizes]
    streams=[torch.npu.current_stream(),torch.npu.Stream()]
    torch.npu.synchronize()
    outputs=[]
    before=read()
    def execute(i):
        bank=banks[i%2]
        for bias,w in bank:
            x=xs[i][w.size(0)]
            if args.kind=="gate":
                x=x.detach().requires_grad_()
                bias=bias.detach().requires_grad_()
                w=w.detach().requires_grad_()
            y=torch.addmm(bias,x,w)
            if args.kind=="gate":
                grad=torch.autograd.grad(y,(x,bias,w),inputs[i][w.size(1)])
                outputs.append([hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
                                 for t in (y,*grad)])
    if args.kind=="gate":
        for i in range(60):
            with torch.npu.stream(streams[i%2]):
                execute(i)
    else:
        for i in range(10):
            execute(i)
        torch.npu.synchronize()
        start=time.perf_counter_ns()
        for i in range(10,60):
            execute(i)
    torch.npu.synchronize()
    us=(time.perf_counter_ns()-start)/50000 if args.kind=="timing" else None
    delta=[a-b for a,b in zip(read(),before)]
    expected=os.environ.get("NPU_DYNAMIC_ADDMM_REBASE_128")=="1"
    if delta[0]!=(480 if expected else 0) or delta[1] or delta[2]!=(960 if expected else 0):
        raise ValueError(f"unexpected fresh/hit path counters {delta}")
    # Exercise a dense but deliberately misaligned region of the very same owner.
    bias,w=banks[0][0]
    off=bias.storage_offset()+1
    bad=flats[0].as_strided(bias.size(),bias.stride(),off)
    invalid=torch.addmm(bad,xs[0][w.size(0)],w)
    fallback=read()[1]-before[1]-delta[1]
    if fallback!=int(expected):
        raise ValueError("misaligned bias did not select the expected original descriptor path")
    data=dict(counters=delta,fallback=fallback,invalid_hash=hashlib.sha256(invalid.cpu().view(torch.uint8).numpy().tobytes()).hexdigest(),
              us_per_eight_addmm=us,words=outputs,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    args.output.write_text(json.dumps(data))


def main(args):
    if args.kind:
        return worker(args)
    data={"gate_exact":False,"timing":[]}
    def invoke(k,route,num):
        path=args.output.parent/f"{k}-{num}-{route}.json"
        cmd=[sys.executable,"-B",__file__,"--layout",str(args.layout),"--schedule",str(args.schedule),
             "--output",str(path),"--kind",k]
        env={**os.environ,"NPU_DYNAMIC_ADDMM_REBASE_128":str(int(route=="B"))}
        subprocess.run(cmd,check=True,env=env)
        return json.loads(path.read_bytes())
    try:
        a=invoke("gate","A",0)
        b=invoke("gate","B",1)
        data["differing_projections"]=[i for i,(x,y) in enumerate(zip(a["words"],b["words"],strict=True)) if x!=y]
        data["gate_exact"]=(len(a["words"])==480 and not data["differing_projections"] and
                            a["invalid_hash"]==b["invalid_hash"] and a["library_sha256"]==b["library_sha256"])
        data["counters"]=[a["counters"],b["counters"]]
        if data["gate_exact"]:
            for i,route in enumerate("ABBA"):
                row=invoke("timing",route,i)
                data["timing"].append(dict(route=route,us_per_eight_addmm=row["us_per_eight_addmm"],counters=row["counters"]))
    finally:
        args.output.write_text(json.dumps(data,indent=2))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for f in ("layout","schedule","output"):
        cli.add_argument("--"+f,required=True,type=Path)
    cli.add_argument("--kind",choices=("gate","timing"))
    main(cli.parse_args())
