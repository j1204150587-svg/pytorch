"""Exercise stock-key equality and real address rebinding, then time independent A/B and B/A processes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def child(a):
    import ctypes
    import torch
    import torch_npu
    torch.npu.set_device(0)
    schedule=json.loads(a.schedule.read_bytes())
    lib=Path(torch_npu.__file__).parent/"lib/libtorch_npu.so"
    count=ctypes.CDLL(str(lib)).npu_div_geometry_count
    count.argtypes=[ctypes.c_uint]
    count.restype=ctypes.c_uint64
    gen=torch.Generator().manual_seed(4314)
    op=torch.ops.aten.div.Scalar
    def new_input(b,l,offset,special=False):
        backing=torch.randn(offset+b*l*64,generator=gen).to(torch.bfloat16)
        if special:
            words=[0,-32768,1,-32767,127,-32641,128,-32640,32639,-129,32640,-128,32705,-63,32641,-127]
            backing.view(torch.int16)[offset:offset+len(words)]=torch.tensor(words,dtype=torch.int16)
        return backing.npu()[offset:].view(b,l,1,64).expand(b,l,4,64)
    report={"mode":a.mode}
    if a.mode in {"audit","bind"}:
        rows=[]
        streams=[torch.npu.current_stream(),torch.npu.Stream()]
        for j,stream in enumerate(streams):
            with torch.npu.stream(stream):
                for i in (range(len(schedule)) if j==0 else [0,4,5,59]):
                    b,l=schedule[i]
                    offset=0 if j==0 else 7
                    # Hold all outputs simultaneously and keep identical geometry while varying live addresses and contents.
                    X=[new_input(b,l,offset,i in (0,4,5,59)) for _ in range(3)]
                    results=[op(x,4) for x in X]
                    refs=[]
                    for x in X:
                        ref=torch.empty(x.shape,device=x.device,dtype=x.dtype)
                        torch.ops.aten.div.out(x,torch.tensor(4),out=ref)
                        refs.append(ref)
                    words=[y.cpu().view(torch.int16) for y in results]
                    ok=all(torch.equal(y,z.cpu().view(torch.int16)) for y,z in zip(words,refs))
                    independent=len({v.untyped_storage().data_ptr() for v in [*X,*results]})==6
                    layout=all(v.is_contiguous() and v.dtype==torch.bfloat16 and v.shape==X[0].shape for v in results)
                    rows.append(dict(shape=[b,l],stream=j,offset=offset,passed=ok and independent and layout))
        x=new_input(2,14,0)
        for t,s in ((x,3),(x.contiguous(),4),(x[:,:,:3],4),(x[:,:,:,:32],4),(new_input(1,8,0),4)):
            y=op(t,s)
            ref=torch.empty_like(y)
            torch.ops.aten.div.out(t,torch.tensor(s),out=ref)
            rows.append(dict(shape=list(t.shape),fallback=True,passed=torch.equal(y.cpu().view(torch.int16),ref.cpu().view(torch.int16))))
        report["cases"]=rows
        report["counters"]=[count(i) for i in range(6)]
        fb,first,reuse,checked,hits,misses=report["counters"]
        report["passed"]=all(r["passed"] for r in rows) and fb>=5 and first>=60 and reuse>=120 and hits>0 and hits+misses==sum(not r.get("fallback",False) for r in rows)*3 and checked==(reuse if a.mode=="audit" else 0)
    else:
        inputs=[new_input(b,l,0) for b,l in schedule]
        for x in inputs:
            _=op(x,4)
        torch.npu.synchronize()
        def window(bank,repeat):
            torch.npu.synchronize()
            begin=time.perf_counter_ns()
            for x in bank:
                for _ in range(repeat):
                    y=op(x,4)
            torch.npu.synchronize()
            return (time.perf_counter_ns()-begin)/(len(bank)*repeat*1000)
        report["us_per_op"]={
            tag:[window(bank,repeats) for _ in range(9)]
            for tag,bank,repeats in [("shape4",[inputs[4]],256),("shape5",[inputs[5]],256),("all60",inputs,16)]}
        report["passed"]=True
    a.output.write_text(json.dumps(report,indent=2))
    if not report["passed"]:
        raise SystemExit(2)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--mode",default="parent")
    a=ap.parse_args()
    if a.mode!="parent":
        child(a)
        return
    report={"schedule_sha256":hashlib.sha256(a.schedule.read_bytes()).hexdigest(),"cases":[]}
    for i,mode in enumerate(("audit","bind","A","B","B","A")):
        env=dict(os.environ,NPU_DIV_GEOMETRY_KEY=str(int(mode!="A")),
                 NPU_DIV_GEOMETRY_AUDIT=str(int(mode=="audit")),NPU_DIV_GEOMETRY_COUNT=str(int(mode in {"audit","bind"})),
                 NPU_DYNAMIC_REBASE_INDICES="1",NPU_DYNAMIC_REBASE_FORWARD="1",ACLNN_CACHE_LIMIT="65536")
        out=a.output.with_name(f"part-{i}-{mode}.json")
        subprocess.run([sys.executable,"-B",__file__,"--schedule",str(a.schedule),"--output",str(out),"--mode",mode],env=env,check=True)
        r=json.loads(out.read_bytes())
        report["cases"].append({"mode":mode,"file":out.name,"passed":r["passed"],
            "counters":r.get("counters"),"median_us":{k:statistics.median(v) for k,v in r.get("us_per_op",{}).items()}})
        a.output.write_text(json.dumps(report,indent=2))
    report["complete"]=True
    a.output.write_text(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
