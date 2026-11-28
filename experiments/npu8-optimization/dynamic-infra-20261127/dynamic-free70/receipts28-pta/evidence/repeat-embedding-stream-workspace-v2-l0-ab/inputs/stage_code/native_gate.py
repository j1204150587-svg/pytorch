"""Run through ordinary ATen autograd dispatch and verify every reused address."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path

import torch
import torch_npu


COUNTERS=("eligible","fallback","prepared","reused","launches","drained","repeat_rejected","rebind_failure",
          "live","peak_slots","peak_workspace_bytes","runtime_fallback")


def read_counters():
    paths={s.split()[-1] for s in Path("/proc/self/maps").read_text().splitlines() if "/libtorch_npu.so" in s}
    if len(paths)!=1:
        raise ValueError(f"expected one loaded PTA library but saw {paths}")
    fn=C.CDLL(paths.pop()).npu_dynamic_repeat_counter
    fn.argtypes=[C.c_uint]; fn.restype=C.c_uint64
    return {name:int(fn(i)) for i,name in enumerate(COUNTERS)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    torch.npu.set_device(0)
    run=torch.ops.aten.embedding_dense_backward.default
    start=read_counters(); groups=[]
    mode=os.getenv("NPU_DYNAMIC_REPEAT_EMBEDDING","0")
    for group_index,(b,l) in enumerate(((1,19),(2,14),(2,22),(4,31),(1,19))):
        pending=[]; reference=[]
        for j,t in enumerate((1,17,33,49,0,16,32,1023,1007)):
            pos=torch.arange(b*l*1024*4)
            bank=((pos*5+pos//3+j+group_index)%29).view(b,l,1024,4)
            ids=bank[:,:,t,:]
            grad=((torch.arange(b*l*4*64)+2*j+group_index)%7-3).view(b,l,4,64).to(torch.bfloat16)
            # Independent owners for every launch, released prior to reading the outputs.
            # No host sync is inserted between submissions to the same stream.
            device_bank=bank.npu(non_blocking=True)
            device_grad=grad.npu(non_blocking=True)
            answer=run(device_grad,device_bank[:,:,t,:],64,-1,False)
            pending.append(answer)
            reference.append(run(grad,ids,64,-1,False))
            del device_bank,device_grad
        torch.npu.synchronize()
        same=[torch.equal(got.cpu(),want) for got,want in zip(pending,reference,strict=True)]
        groups.append(dict(shape=[b,l],exact=same,counters=read_counters()))
        if not all(same):
            break
    pre_fallback=read_counters()
    ids=torch.arange(12).reshape(2,6)%4
    grad=torch.ones((2,6,64),dtype=torch.bfloat16)
    fallback_same=torch.equal(run(grad.npu(),ids.npu(),64,-1,False).cpu(),run(grad,ids,64,-1,False))
    after=read_counters()
    ok=len(groups)==5 and all(all(row["exact"]) for row in groups) and fallback_same
    if mode in ("1","repeat"):
        ok = (ok and pre_fallback["eligible"]-start["eligible"]==45 and
              pre_fallback["reused"]-start["reused"]==30 and pre_fallback["prepared"]-start["prepared"]==15 and
              after["fallback"]-pre_fallback["fallback"]==1 and not after["runtime_fallback"] and
              not after["repeat_rejected"] and not after["rebind_failure"] and after["peak_slots"]<=16)
    elif mode=="fresh":
        ok=(ok and pre_fallback["eligible"]-start["eligible"]==45 and
            pre_fallback["prepared"]-start["prepared"]==45 and
            pre_fallback["reused"]==start["reused"] and after["fallback"]-pre_fallback["fallback"]==1 and
            not after["runtime_fallback"] and not after["repeat_rejected"])
    else:
        ok=ok and start==after
    args.output.write_text(json.dumps(dict(mode=mode,groups=groups,pre_fallback=pre_fallback,
        fallback_exact=fallback_same,start=start,after=after,passed=ok),indent=2))
    return 0 if ok else 1

if __name__=="__main__":
    raise SystemExit(main())
