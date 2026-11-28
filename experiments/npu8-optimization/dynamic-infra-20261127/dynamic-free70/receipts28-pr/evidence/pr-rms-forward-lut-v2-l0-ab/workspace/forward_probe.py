"""Profile and time BF16 RMS forward with the native, stock Triton and planned submitters."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

import torch
import torch_npu
import triton
from triton.backends.ascend import driver
from lut_kernel import lut_forward
import lut_rms as rms


SHAPES = ((1,19), (2,14), (2,22), (4,31))
BINARY_SHA = "99712c27dc9b0d7313fc4096b03c5c0713763a93c186b7a018687b29316064f0"
DRIVER_SHA = "b39118053062f3fc6d7f1047f8c9eb229de47a2b6106d8119cc7087f5b38818e"


def plans(binary, info, fold):
    if list(binary.src.fn.arg_names) != ["x","w","lut","inv_out","z_out","y_out","trace","DEBUG"]:
        return None
    index = lambda key: binary.src.fn.arg_names.index(key) if isinstance(key,str) else key
    constants = {index(k):v for k,v in binary.src.constants.items()}
    signature = {index(k):str(v) for k,v in binary.src.signature.items()}
    meta = fold.values(binary.metadata)
    info["runtime"] = dict(constants=str(constants), signature=signature, binary_hash=binary.hash,
                           name=meta["name"], coalesce_factor=meta["coalesce_factor"], mode=meta["parallel_mode"])
    if signature != {i:"*bf16" for i in range(7)} or constants!={7:False}:
        info["reason"]="actual signature/constexpr differs"
        return None
    image = binary.asm.get("npubin")
    if not isinstance(image,bytes):
        # The Ascend backend uses the image format as the compiled kernel's binary key.
        image = next((v for v in binary.asm.values() if isinstance(v,bytes) and v.startswith(b"\x7fELF")), None)
    info["image_sha256"] = hashlib.sha256(image).hexdigest() if image else None
    if (info["image_sha256"] != BINARY_SHA or meta["name"]!="lut_forward" or meta["parallel_mode"]!="simd"
            or meta["force_simt_only"] or int(meta.get("coalesce_factor") or 1)!=1):
        info["reason"]="kernel hash, mode or grid transformation differs"
        return None
    witness = fold.witness_eager_launcher(binary)
    info["driver"] = witness
    source = driver.generate_npu_wrapper_src(constants,signature,binary.metadata)
    # This is the same source whose content-addressed path must match the loaded
    # JIT launcher's shared object inside witness_eager_launcher().
    args_block = re.search(r"}\s*args\s*=\s*{(.*?)};",source,re.S)
    if not args_block:
        info["reason"]="generated initializer not found"
        return None
    init = args_block.group(1)
    info["generated_argument_initializer"] = init.strip()
    ids = re.findall(r"static_cast<void\*>\(arg(\d+)\)",init)
    caps = re.findall(r"blockNum\s*=\s*std::min\(blockNum,\s*\(uint32_t\)(\d+)\)",source)
    info["physical_caps_in_all_generated_entrypoints"] = caps
    info["logical_product_assignments"] = re.findall(r"uint32_t blockNum\s*=\s*gridX\s*\*\s*gridY\s*\*\s*gridZ",source)
    ok = (witness["driver_py_sha256"]==DRIVER_SHA and witness["auto_map_physical_cap"]==40
          and witness["uses_ffts"] and witness["actual_null_resource_slots"]
          and not witness["allocates_workspace"] and not witness["allocates_sync_lock"]
          and ids==list("0123456") and re.findall(r"\(grid([XYZ])\)",init)==list("XYZ")
          and witness["launcher_source_cache_hash"]=="cdf55e2e9b6509227f1ece00796e4d21c497f7d36a6fc91ca88c24d510d2b8b8"
          and set(caps)=={"40"} and len(caps)==len(info["logical_product_assignments"]))
    if not ok:
        info["reason"]="generated argument order or physical grid/resource guards differ"
        return None
    ext = torch_npu._C
    make = getattr(ext,"_npu_inductor_make_fast_launch_plan",None)
    direct = getattr(ext,"_npu_inductor_fast_launch_with_plan",None)
    capped = getattr(ext,"_npu_hstu_fast_launch_with_physical_cap",None)
    if not callable(make) or not callable(direct):
        info["reason"]="overlay API unavailable"
        return None
    plan = make("lut_forward",binary.function,("tensor",)*7,False,
                int(meta.get("shared_mem_dynamic_size") or 0),False,True)
    plan._owner = binary
    info["capped_entry"] = callable(capped)
    return plan,direct,capped


def sample_stats(values):
    data=sorted(values)
    def rank(f):return data[max(0,(len(data)*f+99)//100-1)]
    return dict(p50=statistics.median(data),p95=rank(95),p99=rank(99),max=data[-1],count=len(data))


def main(args,report):
    sys.path.insert(0,str(args.fold_dir))
    import fold_probe as fold
    from micro import diff
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.ops.load_library(str(args.library.resolve()))
    get_raw = getattr(torch_npu._C,"_npu_getCurrentRawStreamNoWait",None) or torch_npu._C._npu_getCurrentRawStream
    report["environment"] = {k:os.getenv(k) for k in ("TASK_QUEUE_ENABLE","ASCEND_RT_VISIBLE_DEVICES",
            "ASCEND_VISIBLE_DEVICES","TORCHINDUCTOR_NPU_FAST_LAUNCH","TRITON_INTERPRET","TRITON_DEVICE_PRINT")}
    report["torch_npu"] = [torch_npu.__version__,str(torch_npu.__file__)]
    report["stream_getter"]=get_raw.__name__
    gen = torch.Generator().manual_seed(20261127)
    ws = []
    for b,l in SHAPES:
        x = torch.randn((b,l,64),generator=gen).to(torch.bfloat16).npu()
        w = torch.randn(64,generator=gen).to(torch.bfloat16).npu()
        ws.append((x,w))
    def allocate(x,w,table):
        inv=x.new_empty((*x.shape[:-1],1))
        z=torch.empty_like(x)
        y=torch.empty_like(x)
        return x,w,table,inv,z,y,x
    def grid(t):return t[0].numel()//64
    def jit(t):return lut_forward[(grid(t),)](*t,False)
    examples=[allocate(x,w,rms.get_table(x,rms.EPS)) for x,w in ws]
    compiled=[jit(t) for t in examples]
    torch.npu.synchronize()
    if not all(b is compiled[0] for b in compiled):
        raise ValueError("compiled kernel identity changed by shape")
    binary=compiled[0]
    abi=report["plan"]={}
    plan=plans(binary,abi,fold)
    counts=Counter()
    raw_streams={}
    def submit(route,t):
        if route=="S":
            counts["stock"]+=1
            jit(t)
            return
        if plan is None:
            raise ValueError("planned launch attempted without verified ABI")
        if bool(torch.autograd.profiler._is_profiler_enabled) or fold.active_hooks(binary):
            counts["fallback_profiler_or_hooks"]+=1
            jit(t)
            return
        x=t[0]
        g=grid(t)
        if ((x.device.index!=torch.npu.current_device()) or not 0<g<=65535
                or any(v.device!=x.device or v.dtype!=torch.bfloat16 or not v.is_contiguous()
                       or v.data_ptr()%16 for v in t)):
            raise ValueError("layout/device/grid changed from validated input")
        raw=int(get_raw(x.device.index))
        if not raw:raise ValueError("empty current stream")
        raw_streams[raw]=raw_streams.get(raw,0)+1
        o,direct,capped=plan
        if g>40 and not callable(capped):
            counts["fallback_missing_physical_cap_entry"]+=1
            jit(t)
            return
        counts["entered"]+=1
        if g>40:
            capped(o,raw,g,1,1,t,40)
            counts["capped_returned"]+=1
        else:
            direct(o,raw,g,1,1,t)
        counts["returned"]+=1
    def forward(route,inp):
        x,w=inp
        if route=="N":
            return torch.ops.npu_hstu_rms_lut.forward_parts.default(x,w,rms.EPS)
        t=allocate(x,w,rms.get_table(x,rms.EPS))
        submit(route,t)
        return t[5],t[3],t[4]
    routes="NSP" if plan else "NS"
    second=torch.npu.Stream()
    report["gate"]=[]
    for i,inp in enumerate(ws):
        for label,stream in (("default",torch.npu.current_stream()),("alternate",second)):
            torch.npu.synchronize()
            with torch.npu.stream(stream):
                base=forward("N",inp)
                for route in routes[1:]:
                    prior=counts.copy()
                    actual=forward(route,inp)
                    stream.synchronize()
                    checks={name:diff(v,w) for name,v,w in zip(("y","inv","z"),base,actual)}
                    row=dict(shape=SHAPES[i],stream_label=label,raw_stream=int(get_raw(0)),route=route,
                             checks=checks,counts=dict(counts-prior))
                    report["gate"].append(row)
                    if any(r["different"] for r in checks.values()):
                        raise ValueError("forward intermediate mismatch")
    report["gate_counts"]=dict(counts)
    def time_routes(order,mode):
        for route in order:
            host=[]
            before=counts.copy()
            torch.npu.synchronize()
            start=time.perf_counter_ns()
            for i in range(args.repeats):
                j=i%len(ws)
                begin=time.perf_counter_ns()
                if mode=="forward":
                    out=forward(route,ws[j])
                else:
                    submit(route,examples[j])
                host.append((time.perf_counter_ns()-begin)/1000)
            torch.npu.synchronize()
            elapsed=(time.perf_counter_ns()-start)/1000/args.repeats
            report["timing"].append(dict(order=order,route=route,mode=mode,window_us_per_call=elapsed,
                     host_us=sample_stats(host),host_samples_us=host,counts=dict(counts-before)))
    for _ in range(3):
        for inp in ws:
            for route in routes:forward(route,inp)
    report["timing"]=[]
    for order in (routes,routes[::-1],routes,routes[::-1]):
        time_routes(order,"forward")
    if plan:
        for order in ("SP","PS","SP","PS"):
            time_routes(order,"preallocated_kernel")
    if args.profile:
        import torch_npu.profiler as prof
        mark=torch.autograd.profiler.record_function
        torch.npu.synchronize()
        report["profile"]=[]
        with prof.profile(activities=[prof.ProfilerActivity.CPU,prof.ProfilerActivity.NPU],
                on_trace_ready=prof.tensorboard_trace_handler(str(args.profile)),
                experimental_config=prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1)):
            for r in range(3):
                for j,inp in enumerate(ws):
                    b,l=SHAPES[j]
                    for route in (routes if r%2==0 else routes[::-1]):
                        before=counts.copy()
                        tag=f"RMSLUT_{route}_{b}x{l}_r{r}"
                        with mark(tag+"_window"):
                            with mark(tag+"_dispatch"):
                                out=forward(route,inp)
                            torch.npu.synchronize()
                        report["profile"].append(dict(tag=tag,counts=dict(counts-before)))
    report["counts"]=dict(counts)
    report["raw_stream_uses"]=raw_streams
    report["status"]="completed"


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--fold-dir",type=Path,required=True)
    p.add_argument("--repeats",type=int,default=64)
    p.add_argument("--profile",type=Path)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    data={}
    try:
        main(a,data)
    except Exception as e:
        data["error"]=f"{type(e).__name__}: {e}"
        raise
    finally:
        a.out.write_text(json.dumps(data,indent=2)+"\n")
