"""Screen CANN multi-tensor GMM for the two independent linear backward products."""
import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter_ns


def save(path,obj):
    path.parent.mkdir(exist_ok=True,parents=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf8")
    tmp.replace(path)


def word_diffs(a,b):
    import torch
    a,b=a.detach().contiguous().cpu(),b.detach().contiguous().cpu()
    if a.shape!=b.shape or a.dtype!=b.dtype:
        return dict(a=list(a.shape),b=list(b.shape),equal=False)
    bits=(a.view(torch.int16)!=b.view(torch.int16))
    return dict(shape=list(a.shape),equal=not bits.any().item(),different=int(bits.sum()),
                elements=a.numel(),max_abs=(a.float()-b.float()).abs().max().item())


def census(directory):
    files=list(directory.rglob("trace_view.json"))+list(directory.rglob("trace_view.json.gz"))
    if len(files)!=1:
        raise ValueError(f"expected one published trace under {directory}, found {files}")
    fn=files[0]
    with (gzip.open if fn.suffix==".gz" else open)(fn,"rt",encoding="utf-8-sig") as handle:
        raw=json.load(handle)
    events=raw if isinstance(raw,list) else raw["traceEvents"]
    pids={e["args"]["name"]:e["pid"] for e in events if e["name"]=="process_name"}
    spans=[e for e in events if e.get("ph")=="X"]
    tasks=Counter((str(e["args"]["Physic Stream Id"]),e["args"]["Task Type"],e["name"])
                  for e in spans if e["pid"]==pids["Ascend Hardware"] and "Task Type" in e["args"])
    acl=Counter(e["name"] for e in spans if e["pid"]==pids["CANN"] and e["name"].startswith(("AscendCL@aclrtLaunchKernel","AscendCL@aclrtMemcpy")))
    return dict(trace=str(fn),physical=[dict(stream=k[0],kind=k[1],name=k[2],n=v) for k,v in tasks.items()],
                acl=dict(acl),enqueue=dict(Counter(e["name"] for e in spans if e.get("cat")=="enqueue")),
                aten=dict(Counter(e["name"] for e in spans if e["name"].startswith("aten::"))))


def case(args):
    import torch
    torch.set_num_threads(1)
    b,l,cin,cout=args.case
    tag=f"b{b}-l{l}-i{cin}-o{cout}"
    out=args.output/(tag+".json")
    result=dict(shape=[b,l],M=b*l,cin=cin,cout=cout,device=args.device,torch=torch.__version__,
                tq=os.getenv("TASK_QUEUE_ENABLE"),seen_once=False,status="started")
    save(out,result)
    try:
        if args.device=="npu":
            import torch_npu
            torch.npu.set_device(0)
            sync=torch.npu.synchronize
            result["torch_npu"]=torch_npu.__version__
        else:
            sync=lambda:None
        torch.manual_seed(3201+b*l+cin+cout)
        x=torch.randn(b,l,cin).to(device=args.device,dtype=torch.bfloat16).requires_grad_()
        w=torch.randn(cout,cin).to(device=args.device,dtype=torch.bfloat16).requires_grad_()
        bias=torch.randn(cout).to(device=args.device,dtype=torch.bfloat16).requires_grad_()
        dy=torch.randn(b,l,cout).to(device=args.device,dtype=torch.bfloat16)
        y=torch.nn.functional.linear(x,w,bias)
        true_dx,true_dw,_=torch.autograd.grad((y*dy).float().sum(),(x,w,bias))
        flat_x,w=x.detach().view(-1,cin),w.detach()
        flat_dy=dy.view(-1,cout)
        def baseline():
            return flat_dy.mm(w),flat_dy.t().mm(flat_x)
        def grouped():
            return torch_npu.npu_grouped_matmul([flat_dy,flat_dy.t().contiguous()],[w,flat_x],
                                                group_list=None,split_item=0,group_type=-1)
        def compare(outputs):
            return dict(dx=word_diffs(true_dx.reshape(b*l,cin),outputs[0]),dw=word_diffs(true_dw,outputs[1]))
        result["strides"]={k:list(v.stride()) for k,v in
                           (("dy",flat_dy),("dyT",flat_dy.t()),("weight",w),("x",flat_x))}
        result["transpose_copy_bytes"]=flat_dy.numel()*flat_dy.element_size()
        result["stock_vs_manual"]=compare(baseline())
        if args.device=="cpu":
            result["status"]="cpu_reference"
            save(out,result)
            return 0
        sync()
        result["phase"]="first_grouped_call"
        save(out,result)
        tick=perf_counter_ns()
        first=grouped()
        sync()
        result["first_call_us"]=(perf_counter_ns()-tick)/1000
        result["seen_once"]=True
        result["stock_vs_grouped"]=compare(first)
        result["phase"]="separate_profile_or_timing"
        save(out,result)
        if args.profile_case:
            for name,fun in (("A",baseline),("B",grouped)):
                directory=args.output/(tag+"-"+name+"-profile")
                cfg=torch_npu.profiler._ExperimentalConfig(profiler_level=torch_npu.profiler.ProfilerLevel.Level1)
                sync()
                with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],experimental_config=cfg,
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(directory))):
                    with torch.profiler.record_function("HSTULinearBackward"+name):
                        keep=fun()
                    sync()
                result.setdefault("profiles",{})[name]=census(directory)
        if args.repeat:
            times=[]
            for fun in (baseline,grouped):
                for _ in range(5): keep=fun()
                sync()
            for order in ("AB","BA","AB","BA"):
                row={"order":order}
                for name in order:
                    fun=baseline if name=="A" else grouped
                    sync()
                    t=perf_counter_ns()
                    for _ in range(args.repeat): keep=fun()
                    sync()
                    row[name]=(perf_counter_ns()-t)/1e3/args.repeat
                times.append(row)
            result["window_us_per_pair"]=times
            result["median_paired_delta_us"]=statistics.median(row["A"]-row["B"] for row in times)
        result["status"]="completed"
        save(out,result)
        return 0
    except Exception as exc:
        result.update(status="failed",error=type(exc).__name__+": "+str(exc)[:4500])
        save(out,result)
        return 1


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--device",choices=["npu","cpu"],required=True)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--schedule",type=Path)
    cli.add_argument("--case",type=int,nargs=4,metavar=("B","L","IN","OUT"))
    cli.add_argument("--repeat",type=int,default=32)
    cli.add_argument("--profile-case",action="store_true")
    cli.add_argument("--all",action="store_true",help="after screening include every scheduled shape")
    args=cli.parse_args()
    if args.case:
        return case(args)
    scheduled=[tuple(pair) for pair in json.loads(args.schedule.read_text())]
    selected=[(2,16),(1,19),(2,14),(2,22)]
    if not all(v in scheduled for v in selected):
        raise ValueError("screening shapes missing from actual schedule")
    if args.all: selected.extend(v for v in scheduled if v not in selected)
    # Isolate each unsupported ACL call. A failed asynchronous launch can
    # poison a device context, so it must never supply a later control result.
    summary=[]
    for b,l in selected:
        for ci,co in ((64,256),(192,64)):
            command=[sys.executable,__file__,"--device",args.device,"--output",str(args.output),
                     "--case",str(b),str(l),str(ci),str(co),"--repeat",str(args.repeat)]
            if args.profile_case and (b,l) in ((2,16),(1,19)):
                command.append("--profile-case")
            res=subprocess.run(command)
            f=args.output/f"b{b}-l{l}-i{ci}-o{co}.json"
            receipt=json.loads(f.read_text()) if f.exists() else {"status":"missing"}
            summary.append(dict(shape=[b,l],cin=ci,cout=co,returncode=res.returncode,status=receipt["status"]))
            save(args.output/"summary.json",summary)
    return 0 if all(x["returncode"]==0 for x in summary) else 1


if __name__=="__main__":
    sys.exit(main())
