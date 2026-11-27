"""Independent generated-wrapper microbenchmark; synchronized AB/BA, fresh caches, no profiler."""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
from time import perf_counter_ns
import traceback
from types import ModuleType

from compiled_control import SHAPES, files_under


def pointwise(x, y):
    return x * 2.0 + y


def save(path, data):
    tmp = path.with_suffix(path.suffix+".partial")
    tmp.write_text(json.dumps(data,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_pointwise():
    paths = [Path(__file__),Path(__file__).with_name("compiled_control.py")]
    functions = []
    for p in paths:
        tree = ast.parse(p.read_bytes())
        fn, = [n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=="pointwise"]
        functions.append([ast.dump(line,include_attributes=False) for line in fn.body])
    if functions[0] != functions[1]:
        raise ValueError("operator differs from compiled_control.py")


def pct(samples, q):
    return sorted(samples)[math.ceil(len(samples)*q/100)-1]


def describe(samples):
    return {"count":len(samples),"mean_us":statistics.fmean(samples)/1000,
            **{f"p{n}_us":pct(samples,n)/1000 for n in (50,95,99,100)}}


def postflight(torch, compiled, data, cache, switch):
    # All inspection/counters start after timing. The original selected C
    # callables are taken from the already-promoted generated wrapper slots.
    from torch._inductor.codecache import PyCodeCache
    modules = [m for m in PyCodeCache.modules if isinstance(m,ModuleType)]
    modules += [m for m in tuple(sys.modules.values()) if isinstance(m,ModuleType) and
                str(vars(m).get("__file__","") or "").startswith(str(cache))]
    seen = set()
    plan_states,patches = [],[]
    import torch_npu
    native = torch_npu._C._npu_inductor_fast_launch_with_plan
    rows = []
    step_hits = []
    for m in modules:
        if id(m) in seen:
            continue
        seen.add(id(m))
        for name,slot in list(vars(m).items()):
            if not name.startswith("_npu_fast_launch_call_") or not isinstance(slot,list) or not slot:
                continue
            bound = slot[0]
            direct = getattr(bound,"_direct",None)
            launch = getattr(direct,"untimed_launch",None)
            rep = {"slot":name,"bound":type(bound).__name__,"promoted":direct is not None,
                   "raw_cpp_callable":launch is native,"wrapper":str(getattr(m,"__file__","")),
                   "kernel":getattr(getattr(direct,"launcher",None),"_npu_fast_launch_kernel_name",None)}
            plan_states.append(rep)
            if direct is None or launch is not native or any(direct is p[0] for p in patches):
                continue
            patches.append((direct,launch))
            def capture(plan,stream,*a,_orig=launch,_kernel=rep["kernel"],**kw):
                result = _orig(plan,stream,*a,**kw)
                step_hits.append({"kernel":_kernel,"stream_handle":int(stream),"grid":[int(x) for x in a[:3]]})
                return result
            direct.untimed_launch = capture
    try:
        for shape,x,y,expected in data:
            step_hits.clear()
            actual = compiled(x,y)
            torch.npu.synchronize()
            rows.append({"shape":shape,"exact":torch.equal(actual.cpu(),expected),"planned":list(step_hits)})
    finally:
        for direct,original in patches:
            direct.untimed_launch = original
    enabled = switch=="1"
    passed = (all(r["exact"] for r in rows)
              and all(len(r["planned"])==(1 if enabled else 0) for r in rows)
              and (bool(patches) if enabled else not plan_states))
    return {"slots":plan_states,"calls":rows,"verified":passed}


def worker(args, r):
    cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).resolve()
    for name in ("TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR","XDG_CACHE_HOME"):
        path = Path(os.environ[name])
        if not path.is_dir() or any(path.iterdir()):
            raise ValueError("expected newly created empty cache: "+name)
    import torch
    import torch_npu
    torch.npu.set_device(0)
    switch = os.environ["TORCHINDUCTOR_NPU_FAST_LAUNCH"]
    shapes = list(SHAPES) if args.mode=="dynamic" else [tuple(args.static_shape)]
    cache_vars = ("TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR","XDG_CACHE_HOME","ASCEND_CACHE_PATH",
                  "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE","TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE")
    r.update(pid=os.getpid(),mode=args.mode,switch=switch,shapes=shapes,torch=torch.__version__,
             torch_npu=torch_npu.__version__,package=str(torch_npu.__file__),python=sys.version,
             symbolic_group_autotune=os.environ.get("INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE"),
             platform=platform.platform(),caches={k:os.environ.get(k) for k in cache_vars},
             timing={"warmup_per_shape":args.warmup,"calls":args.calls,"raw_host_return_ns":[]},
             source_sha256=digest(Path(__file__)),control_sha256=digest(Path(__file__).with_name("compiled_control.py")))
    if torch.autograd.profiler._is_profiler_enabled or sys.getprofile() or sys.gettrace():
        raise ValueError("instrumentation is active")
    check_pointwise()
    from compiled_runtime_compat import install as install_compiler_guard
    print("PR compiler guard", install_compiler_guard(), flush=True)
    compiled = torch.compile(pointwise,backend="inductor",dynamic=args.mode=="dynamic",fullgraph=True)
    data = []
    for b,l in shapes:
        v = torch.arange(b*l*64).reshape(b,l,64)
        x,y = (v%7-3).bfloat16(), ((v//3)%9-4).bfloat16()
        data.append(((b,l),x.npu(),y.npu(),pointwise(x,y)))
    first = []
    for shape,x,y,expected in data:
        torch.npu.synchronize()
        start = perf_counter_ns()
        actual = compiled(x,y)
        torch.npu.synchronize()
        first.append({"shape":shape,"compile_or_first_sync_ms":(perf_counter_ns()-start)/1e6,
                      "exact":torch.equal(actual.cpu(),expected)})
    r["first_shape"] = first
    for _ in range(args.warmup):
        for _,x,y,_ in data:
            last = compiled(x,y)
    torch.npu.synchronize()
    wrapper_before = files_under(cache)
    seq = [data[i%len(data)] for i in range(args.calls)]
    samples = []
    if torch.autograd.profiler._is_profiler_enabled or sys.getprofile() or sys.gettrace():
        raise ValueError("instrumentation active at window start")
    torch.npu.synchronize()
    t0 = perf_counter_ns()
    for _,x,y,_ in seq:
        st = perf_counter_ns()
        last = compiled(x,y)
        samples.append(perf_counter_ns()-st)
    submitted = perf_counter_ns()
    torch.npu.synchronize()
    end = perf_counter_ns()
    timing = r["timing"]
    timing.update(raw_host_return_ns=samples,host=describe(samples),window_wall_ns=end-t0,
                  sync_window_us_per_call=(end-t0)/1000/len(samples),final_sync_drain_us=(end-submitted)/1000,
                  per_shape_host={"x".join(map(str,s)):describe(samples[j::len(shapes)]) for j,s in enumerate(shapes)})
    r["last_exact"] = torch.equal(last.cpu(),seq[-1][3])
    r["wrappers"] = files_under(cache)
    r["no_new_wrapper_in_window"] = r["wrappers"] == wrapper_before
    save(args.worker,r)
    r["postflight"] = postflight(torch,compiled,data,cache,switch)
    r["valid"] = (r["postflight"]["verified"] and r["last_exact"] and all(x["exact"] for x in first)
                  and bool(r["wrappers"]) and r["no_new_wrapper_in_window"])


def percent(a,b):
    return (1-b/a)*100


def control_links(trace, controls):
    evidence = {}
    if trace:
        raw = json.loads(trace.read_text())
        evidence["same_tasks_and_streams"] = raw.get("same_tasks_and_streams",False)
        evidence["trace_shapes"] = [r["shape"] for r in raw.get("a",[])]
        evidence["trace"] = {"file":str(trace.resolve()),"sha256":digest(trace)}
    if controls:
        evidence["activation"] = []
        for p in controls:
            raw = json.loads(p.read_text())
            rows = [r for r in raw.get("steps",[]) if r["phase"] in ("after","revisit")]
            traced = [r for r in raw.get("steps",[]) if r["phase"] in ("level0","level1")]
            evidence["activation"].append({"file":str(p.resolve()),"sha256":digest(p),"switch":raw.get("switch"),
                "successful_plans_unprofiled":[r["counts"].get("planned_submit_ok",0) for r in rows],
                "successful_plans_during_trace":[r["counts"].get("planned_submit_ok",0) for r in traced],
                "shapes_during_trace":[r["shape"] for r in traced],
                "profile_requested":raw.get("profile_requested"),"error":raw.get("error")})
    # The stock compiled_control generator is dynamic. The parent report keeps
    # its hardware proof separate from a static specialization.
    return evidence


def summarize(out,trace,controls):
    index = json.loads((out/"index.json").read_text())
    pairs = []
    for i,order in enumerate(index["orders"]):
        records = {}
        for leg in order:
            p = out/f"{i:02d}-{leg}"/"result.json"
            if p.exists():
                records[leg] = json.loads(p.read_text())
        if len(records)!=2 or not all(r.get("valid") for r in records.values()):
            pairs.append({"order":order,"valid":False})
            continue
        a,b = records["A"]["timing"],records["B"]["timing"]
        pairs.append({"order":order,"valid":True,
                      "sync_window_us_per_call":{"A":a["sync_window_us_per_call"],"B":b["sync_window_us_per_call"]},
                      "sync_window_reduction_pct":percent(a["sync_window_us_per_call"],b["sync_window_us_per_call"]),
                      "host_A":a["host"],"host_B":b["host"],
                      "host_quantile_reduction_pct":{k:percent(a["host"][k],b["host"][k]) for k in
                                                     ("p50_us","p95_us","p99_us")}})
    refs = control_links(trace,controls)
    controls_ok = len(refs.get("activation",[]))==2 and all(
        c["switch"]==str(leg) and c["error"] is None and len(c["successful_plans_unprofiled"])==8
        and all(n==leg for n in c["successful_plans_unprofiled"])
        and c["successful_plans_during_trace"] == [leg]*len(SHAPES)
        and c["shapes_during_trace"] == [list(s) for s in SHAPES]
        for leg,c in enumerate(refs["activation"]))
    task_proof = (index["mode"]=="dynamic" and controls_ok and refs.get("same_tasks_and_streams") is True
                  and refs.get("trace_shapes")==[f"{b}x{n}" for b,n in SHAPES])
    result = {"workload":"independent compiled x * 2.0 + y; forward only", "mode":index["mode"],
              "matching_control_proves_same_device_tasks_and_streams":task_proof,
              "pairs":pairs,"external_evidence":refs,"all_valid":all(p["valid"] for p in pairs)}
    save(out/"summary.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result["all_valid"]


def orchestrate(args):
    out = args.output.resolve()
    if args.summarize:
        return 0 if summarize(out,args.trace_comparison,args.control_receipts) else 1
    if out.exists() and any(out.iterdir()):
        raise ValueError("use a new output directory")
    out.mkdir(parents=True,exist_ok=True)
    orders = args.orders
    save(out/"index.json",dict(orders=orders,mode=args.mode,warmup=args.warmup,calls=args.calls,
                               static_shape=args.static_shape,scope="standalone generated-wrapper microbenchmark"))
    for i,order in enumerate(orders):
        for leg in order:
            folder = out/f"{i:02d}-{leg}"
            folder.mkdir()
            env = dict(os.environ)
            env["TORCHINDUCTOR_NPU_FAST_LAUNCH"] = str(int(leg=="B"))
            env["INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE"] = "0"
            env["TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE"] = "0"
            env["TORCHINDUCTOR_AUTOTUNE_REMOTE_CACHE"] = "0"
            for key in ("TORCHINDUCTOR_CACHE_DIR","TRITON_CACHE_DIR","XDG_CACHE_HOME","ASCEND_CACHE_PATH"):
                path = folder/key.lower()
                path.mkdir()
                env[key] = str(path)
            cmd = [sys.executable,"-B",str(Path(__file__).resolve()),"--worker",str(folder/"result.json"),
                   "--mode",args.mode,"--warmup",str(args.warmup),"--calls",str(args.calls),
                   "--static-shape",*map(str,args.static_shape)]
            save(folder/"command.json",{"argv":cmd,"turn":leg,"order":order,"toggle":env["TORCHINDUCTOR_NPU_FAST_LAUNCH"]})
            print("Running",folder.name,flush=True)
            with (folder/"stdout.log").open("wb") as log:
                c = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
            if c.returncode:
                summarize(out,args.trace_comparison,args.control_receipts)
                return c.returncode
    return 0 if summarize(out,args.trace_comparison,args.control_receipts) else 1


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output",type=Path)
    cli.add_argument("--worker",type=Path,help=argparse.SUPPRESS)
    cli.add_argument("--mode",choices=("static","dynamic"),default="dynamic")
    cli.add_argument("--static-shape",type=int,nargs=2,default=(1,19))
    cli.add_argument("--orders",nargs="+",choices=("AB","BA"),default=("AB","BA"))
    cli.add_argument("--warmup",type=int,default=20,help="untimed calls per shape after the first call")
    cli.add_argument("--calls",type=int,default=512)
    cli.add_argument("--summarize",action="store_true")
    cli.add_argument("--trace-comparison",type=Path)
    cli.add_argument("--control-receipts",type=Path,nargs=2,metavar=("A","B"))
    args = cli.parse_args()
    if args.calls<4 or args.warmup<1 or min(args.static_shape)<1:
        cli.error("require at least four calls, one warmup and positive dimensions")
    if args.worker:
        report = {"valid":False}
        try:
            worker(args,report)
        except Exception:
            report["error"] = traceback.format_exc()
        save(args.worker,report)
        print(json.dumps({k:report.get(k) for k in ("valid","error")}),flush=True)
        sys.exit(0 if report["valid"] else 1)
    if not args.output:
        cli.error("provide --output")
    sys.exit(orchestrate(args))
