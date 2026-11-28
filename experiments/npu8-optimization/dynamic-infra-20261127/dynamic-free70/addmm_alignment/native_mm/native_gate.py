"""Validate the real op-plugin Mm backend on independent BF16 parameter owners and streams."""
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
    library = Path(torch_npu.__file__).parent / "lib/libtorch_npu.so"
    cmd = ctypes.CDLL(str(library)).npu_mm_rebase_counter
    cmd.argtypes = [ctypes.c_uint]
    cmd.restype = ctypes.c_uint64
    counts = lambda: [cmd(i) for i in range(4)]
    def digest(t):
        return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    shapes = json.loads(args.schedule.read_bytes())
    layout = json.loads(args.layout.read_bytes())
    g = torch.Generator().manual_seed(1234)
    def values(*shape):
        return torch.randn(*shape,generator=g).bfloat16().npu()
    owners = [values(layout["flat"]["elements"]) for _ in range(2)]
    banks = []
    for owner in owners:
        bank = []
        for layer in reversed(layout["projection_views"]):
            view = layer["transposed_weight"]
            w = owner.as_strided(view["shape"],view["stride"],view["storage_element_offset"]).t()
            assert w.is_contiguous() and w.data_ptr() % 128 == 0
            bank.append(w)
        banks.append(bank)
    inputs = [{k:values(b*l,k) for k in (64,256)} for b,l in shapes]
    upstream = [{k:values(b*l,k) for k in (64,192)} for b,l in shapes]
    streams = (torch.npu.current_stream(),torch.npu.Stream())
    torch.npu.synchronize()
    outputs = []
    def step(i):
        for w in banks[i%2]:
            x = inputs[i][w.size(0)]
            if args.mode == "gate":
                x = x.detach().requires_grad_()
                w = w.detach().requires_grad_()
            y = torch.mm(x,w)
            if args.mode == "gate":
                dx,dw = torch.autograd.grad(y,(x,w),upstream[i][w.size(1)])
                outputs.append([digest(t) for t in (y,dx,dw)])
    start_counts = counts()
    if args.mode == "gate":
        for i in range(60):
            with torch.npu.stream(streams[i%2]):
                step(i)
        us = None
    else:
        for i in range(10):
            step(i)
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        for i in range(10,60):
            step(i)
        torch.npu.synchronize()
        us = (time.perf_counter_ns()-start)/50000
    seen = counts()
    delta = [v-o for v,o in zip(seen,start_counts)]
    expected = os.environ["NPU_DYNAMIC_MM_REBASE_128"] == "1"
    if delta[0] != (480 if expected else 0) or delta[2] != delta[0]:
        raise ValueError(f"functional Mm scope or cache bindings mismatch {delta}")
    # Same real owner, source shifted by one BF16 element.
    w = banks[0][0]
    invalid = owners[0].as_strided(w.shape,w.stride(),w.storage_offset()+1)
    fallback_out = torch.mm(inputs[0][w.size(0)],invalid)
    last = counts()
    if last[0] != seen[0] or last[1]-seen[1] != int(expected):
        raise ValueError("unaligned weight must select the original descriptor")
    result = dict(library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(), values=outputs,
                  counters=delta, fallback=int(expected), fallback_hash=digest(fallback_out), us_per_eight_mm=us)
    args.output.write_text(json.dumps(result))


def main(args, report):
    def invoke(mode,leg,i):
        out = args.output.with_name(f"{mode}-{i}-{leg}.json")
        argv = [sys.executable,"-B",__file__,"--mode",mode,"--schedule",str(args.schedule),
                "--layout",str(args.layout),"--output",str(out)]
        subprocess.run(argv,env={**os.environ,"NPU_DYNAMIC_MM_REBASE_128":str(int(leg=="B"))},check=True)
        return json.loads(out.read_bytes())
    a,b = [invoke("gate",leg,i) for i,leg in enumerate("AB")]
    report["exact"] = (len(a["values"])==480 and a["values"]==b["values"]
                       and a["fallback_hash"]==b["fallback_hash"] and a["library_sha256"]==b["library_sha256"])
    report["gate_counters"] = [a["counters"],b["counters"]]
    if report["exact"]:
        for i,leg in enumerate("ABBA"):
            row = invoke("timing",leg,i)
            report["windows"].append(dict(leg=leg,us=row["us_per_eight_mm"],counters=row["counters"]))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for key in ("schedule","layout","output"):
        cli.add_argument("--"+key,type=Path,required=True)
    cli.add_argument("--mode",choices=("gate","timing"))
    arguments = cli.parse_args()
    if arguments.mode:
        worker(arguments)
    else:
        record = dict(exact=False,windows=[])
        try:
            main(arguments,record)
        finally:
            arguments.output.write_text(json.dumps(record,indent=2))
