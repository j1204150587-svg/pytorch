"""N/S/P paired gate and synchronized windows with two real Ascend streams."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback

import torch
import torch_npu


def run(a,out):
    sys.path.extend(map(str,[a.fold_dir,a.attention_dir,a.scores_dir]))
    import locked as lock
    import operator_impl as impl
    from micro import bits,build,execute,original
    impl.launch=lock.launch
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    funcs={"N":original,"S":lambda q,k,v:impl.apply(q@k.transpose(-2,-1),False)@v,
          "P":lambda q,k,v:impl.apply(q@k.transpose(-2,-1),True)@v}
    streams=(torch.npu.current_stream(),torch.npu.Stream())
    saved=[]
    for i,(b,l) in enumerate(json.loads(a.schedule.read_bytes())):
        tags=("finite","signed_zero","nonfinite") if i in (0,1,3) else ("finite",)
        for case in tags:
            with torch.npu.stream(streams[i%2]):
                data=build(b,l,"npu",case,250+i)
                ref,jit,plan=(execute(funcs[r],data) for r in "NSP")
                def check(left,right):
                    return {n:bits(x,y) for n,x,y in zip(("output","dQ","dK","dV"),left,right,strict=True)}
                row={"shape":[b,l],"case":case,"stream":i%2,"NS":check(ref,jit),"SP":check(jit,plan)}
                if i in (0,1,3,4,5,6):
                    raw=(data[0]@data[1].transpose(-2,-1)).detach().requires_grad_()
                    up=torch.linspace(-2.,2.,raw.numel()).reshape(raw.shape).bfloat16().to(raw.device)
                    s=[impl.stock(raw),impl.apply(raw,False),impl.apply(raw,True)]
                    gradients=[torch.autograd.grad(t,raw,up)[0] for t in s]
                    for label,j in (("NS",0),("SP",1)):
                        row[label+"_score"]={"score":bits(s[j],s[j+1]),"dScore":bits(gradients[j],gradients[j+1])}
                out["cases"].append(row)
                if case=="finite" and [b,l] in ([1,19],[2,14],[2,22],[4,31]):
                    saved.append(data)
    torch.npu.synchronize()
    def equal(prefix):
        return all(v["differing_bf16_words"]==0 for row in out["cases"] for key,group in row.items()
                   if key in (prefix,prefix+"_score") for v in group.values())
    out["stock_matches_N"],out["plan_matches_S"]=equal("NS"),equal("SP")
    if len(out["cases"])!=66 or len(saved)!=4:raise ValueError("incomplete shape gate")
    # Preserve native vs stock arithmetic failures separately from the launcher's
    # strict S/P equality. Windows in that case are diagnostic only.
    if out["plan_matches_S"]:
        for order in ("NSP","PSN","NSP","PSN"):
            window={"order":order}
            for route in order:
                for sample in saved:execute(funcs[route],sample)
                torch.npu.synchronize()
                before=Counter(lock.COUNTS)
                start=time.perf_counter_ns()
                for j in range(48):execute(funcs[route],saved[j%4])
                torch.npu.synchronize()
                window[route]={"us":(time.perf_counter_ns()-start)/48000,"calls":dict(lock.COUNTS-before)}
            out["windows"].append(window)
    out["profile"]=[]
    if a.profile and out["plan_matches_S"]:
        import torch_npu.profiler as profiler
        from torch.autograd.profiler import record_function
        with profiler.profile(activities=[profiler.ProfilerActivity.CPU,profiler.ProfilerActivity.NPU],
             on_trace_ready=profiler.tensorboard_trace_handler(str(a.output.parent/"profile")),
             experimental_config=profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level1)):
            for i,sample in enumerate(saved):
                for route in "SP":
                    before=Counter(lock.COUNTS)
                    with record_function(f"SCORELOCK_{route}_{i}"):
                        execute(funcs[route],sample)
                    torch.npu.synchronize()
                    out["profile"].append({"index":i,"route":route,"counts":dict(lock.COUNTS-before)})
    out["plans"]=lock.EVIDENCE
    out["calls"]=dict(lock.COUNTS)
    out["actual_alloc_init_kernel_returns"]=lock.service_counts()
    out["all_locked_kernel_returns_match"] = all(
        c==[lock.COUNTS["planned_"+name]]*3 for name,c in out["actual_alloc_init_kernel_returns"].items())
    out["complete"]=(out["all_locked_kernel_returns_match"] and not lock.COUNTS["fallback"]
                     and "causal_score_forward" in out["actual_alloc_init_kernel_returns"])
    if out["complete"] and out["plan_matches_S"] and out["stock_matches_N"]:
        from streams import run as stress
        out["concurrent"]={}
        stress(a,out["concurrent"])
        out["complete"]=out["complete"] and out["concurrent"]["accepted"]


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for key in ("fold-dir","scores-dir","attention-dir","schedule","output"):
        cli.add_argument("--"+key,type=Path,required=True)
    cli.add_argument("--profile",action="store_true")
    args=cli.parse_args()
    report={"cases":[],"windows":[],"complete":False}
    try:
        run(args,report)
    except Exception:
        report["error"]=traceback.format_exc()
        raise
    finally:
        args.output.write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k not in ("plans","cases")}),flush=True)
