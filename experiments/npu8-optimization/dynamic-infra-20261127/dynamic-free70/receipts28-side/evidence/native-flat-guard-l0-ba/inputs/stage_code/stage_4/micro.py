"""Compare the Python and actual compiled metadata guards on real tensors without running training."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode
from adapter import MetadataGuard


def oracle(flat,grads,shapes,sizes,verify_storage=True):
    if len(grads)!=len(shapes):
        return None
    root=flat._base if flat._base is not None else flat
    flat_storage=flat.untyped_storage()._cdata if verify_storage else None
    offset=flat.storage_offset()
    for g,s,n in zip(grads,shapes,sizes,strict=True):
        if (type(g) is not torch.Tensor or g._base is not root or (verify_storage and g.untyped_storage()._cdata!=flat_storage)
                or g.device!=flat.device or g.dtype!=flat.dtype or
                g.requires_grad or g.is_neg() or g.is_conj() or g.shape!=s or not g.is_contiguous() or
                g.storage_offset()!=offset):
            return None
        offset+=n
    return flat


class ForwardMode(TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        return func(*args,**(kwargs or {}))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("build",type=Path)
    cli.add_argument("output",type=Path)
    cli.add_argument("--device",choices=("cpu","npu"),default="cpu")
    args=cli.parse_args()
    if args.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    torch.set_num_threads(1)
    shapes=[torch.Size((2,4))]*1052
    sizes=[8]*len(shapes)
    flat=torch.empty(sum(sizes)+31,device=args.device,dtype=torch.bfloat16).narrow(0,7,sum(sizes))
    good=[v.view(s) for v,s in zip(flat.split(sizes),shapes)]
    original=lambda f,g:oracle(f,g,shapes,sizes)
    native=MetadataGuard(shapes,sizes,enabled=True,directory=args.build)
    py=MetadataGuard(shapes,sizes,enabled=False)
    def replaced(x):
        g=list(good);g[500]=x;return g
    swapped=list(good);swapped[20],swapped[21]=swapped[21],swapped[20]
    changed_storage=good[500].as_strided(good[500].shape,good[500].stride(),good[500].storage_offset())
    with torch.no_grad():
        foreign=torch.empty_like(flat._base)
        changed_storage.set_(foreign.untyped_storage(),good[500].storage_offset(),(2,4),(4,1))
    cases={"contiguous":good,"tuple":tuple(good),"short":good[:-1],"swapped":swapped,
           "shape":replaced(good[500].reshape(4,2)),"layout_offset":replaced(good[501]),
           "noncontiguous":replaced(good[500].reshape(4,2).t()),
           "detached":replaced(good[500].detach()),"wrong_root":replaced(foreign[:8].reshape(2,4)),
           "dtype":replaced(good[500].float()),"parameter":replaced(torch.nn.Parameter(good[500])),
           "requires_grad":replaced(good[500].view_as(good[500]).requires_grad_()),
           "non_tensor":replaced(None),"same_base_changed_storage":replaced(changed_storage)}
    checks={}
    for k,g in cases.items():
        before=native.record["native_accept"]
        expected=original(flat,g)
        actual=native.evaluate(flat,g,original)
        assert actual is expected,k
        checks[k]={"eligible":expected is flat,"native_accepted":native.record["native_accept"]>before}
    rebind=cases["same_base_changed_storage"]
    legacy_rebind=oracle(flat,rebind,shapes,sizes,verify_storage=False) is flat
    assert original(flat,rebind) is None and not checks["same_base_changed_storage"]["native_accepted"]
    with ForwardMode():
        assert native.evaluate(flat,good,original) is flat
    assert native.record["context_fallback"]==1
    # Count any recorded ATen operation *in* successful native validation, separately from construction.
    with torch.autograd.profiler.profile(use_device=None) as profile:
        for _ in range(5): assert native.evaluate(flat,good,original) is flat
    op_counts={e.key:e.count for e in profile.key_averages() if e.key.startswith("aten::")}
    assert not op_counts,op_counts
    if args.device=="npu":torch.npu.synchronize()
    times={"A":[],"B":[]}
    for i in range(58):
        pair=[("A",py),("B",native)]
        for label,obj in (pair if i%2==0 else reversed(pair)):
            start=time.perf_counter_ns()
            assert obj.evaluate(flat,good,original) is flat
            dt=(time.perf_counter_ns()-start)/1e3
            if i>=8:times[label].append(dt)
    quant=lambda v: {"median":statistics.median(v),"p95":sorted(v)[int(.95*(len(v)-1))],"min":min(v),"max":max(v)}
    out={"torch":torch.__version__,"device":args.device,"shards":len(good),"checks":checks,
         "legacy_python_rebound_same_base_accepted":legacy_rebind,
         "mode_fallback":True,"aten_op_counts_in_five_native_scans":op_counts,
         "timings_us":{k:quant(v) for k,v in times.items()},"per_call_us":times,"counters_B":native.record}
    args.output.write_text(json.dumps(out,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:out[k] for k in ("torch","device","shards","timings_us","aten_op_counts_in_five_native_scans")}))


if __name__=="__main__":
    main()
