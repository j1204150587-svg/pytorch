"""Inspect native/Inductor availability and count real plan submissions in one process."""
import argparse
from collections import Counter
import importlib
import json
import os
from pathlib import Path
import sys
import traceback


def run(args, report):
    import torch
    import torch_npu
    report.update(torch=torch.__version__,torch_npu=torch_npu.__version__,package=torch_npu.__file__,
                  pr_toggle=os.getenv("TORCHINDUCTOR_NPU_FAST_LAUNCH"),
                  torch_npu_git=getattr(getattr(torch_npu,"version",None),"git_version",None))
    ext = torch_npu._C
    names = ("_NPUInductorFastLaunchPlan","_npu_inductor_make_fast_launch_plan","_npu_inductor_fast_launch_with_plan")
    report["C_symbols"] = {name:hasattr(ext,name) for name in names}
    if args.kind == "inventory":
        return
    if not all(report["C_symbols"].values()):
        raise RuntimeError("incomplete installed native bindings")
    if args.kind == "abi":
        # Plan creation with no FFTS request allocates only host metadata.
        # Never submit the dummy stub or pass it to the runtime launcher.
        obj = getattr(ext,names[1])("host_metadata_only",1,["tensor","i32","f32"])
        report["host_plan_type_ok"] = isinstance(obj,getattr(ext,names[0]))
        return
    counts = Counter()
    for name in names[1:]:
        original = getattr(ext,name)
        def counted(*a,_orig=original,_name=name,**kw):
            counts[_name + "_attempts"] += 1
            val = _orig(*a,**kw)
            counts[_name + "_success"] += 1
            return val
        setattr(ext,name,counted)
    def fn(x,y):
        return (x+y)*2
    if args.kind in ("frontend","static","dynamic"):
        importlib.import_module("torch_npu._inductor")
        patch = importlib.import_module("torch_npu._inductor.experimental.python_wrapper_fast_launch.patch")
        report["frontend_patched"] = patch._PATCHED
        if args.kind == "frontend":
            return
    torch.npu.set_device(0)
    wrapped = torch.compile(fn,backend="inductor",dynamic=args.kind=="dynamic",fullgraph=True) if args.kind in ("static","dynamic") else fn
    shapes = [2048,2048,2048,2048]*2 if args.kind != "dynamic" else [2048,1536,2048,4096,1536,2048,2048,1536]
    steps = []
    report["steps"] = steps
    for i,n in enumerate(shapes):
        gen = torch.Generator().manual_seed(1004+i)
        x,y = [torch.randn(n,generator=gen,dtype=torch.bfloat16).to("npu") for _ in range(2)]
        ref = fn(x,y)
        before = counts.copy()
        actual = wrapped(x,y)
        torch.npu.synchronize()
        step = {"size":n, "bitwise_output_equal":torch.equal(ref.cpu().view(torch.int16),actual.cpu().view(torch.int16)),
                "counters":dict(counts-before)}
        steps.append(step)
        if not step["bitwise_output_equal"]:
            raise RuntimeError("output differs")
    report["counts"] = dict(counts)
    report["successful_planned_launches"] = counts[names[2]+"_success"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind",choices=("inventory","abi","frontend","eager","static","dynamic"))
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    r = {"kind":args.kind,"complete":False}
    try:
        run(args,r)
        r["complete"] = True
    except Exception:
        r["error"] = traceback.format_exc()
    args.output.write_text(json.dumps(r,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(r),flush=True)
    sys.exit(0 if r["complete"] else 1)
