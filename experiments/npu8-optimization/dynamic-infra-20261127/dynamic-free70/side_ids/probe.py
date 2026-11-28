"""Check eager / Triton / C++ launches on dynamic IDs, then time and profile identical inputs."""
import argparse
from collections import Counter
from contextlib import nullcontext
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_npu
import side_ids_impl as impl

SHAPES=((1,19),(2,14),(2,22),(4,31))


def diff(a,b):
    a,b=a.cpu(),b.cpu()
    if a.shape!=b.shape or a.dtype!=b.dtype:return {"shape_or_dtype":True}
    changed=(a!=b).reshape(-1).nonzero().flatten()
    if not changed.numel():return {}
    k=int(changed[0])
    return {"n":changed.numel(),"first":k,"actual":int(a.reshape(-1)[k]),"expected":int(b.reshape(-1)[k])}


def inputs(b,l,dtype,gen,boundary=True):
    ids=torch.randint(0,64,(b,l,1024,4),dtype=dtype,generator=gen)
    offsets=torch.randperm(1024,generator=gen,dtype=torch.int64).mul_(64).view(1024,1,1,1)
    if boundary:
        v=torch.iinfo(dtype)
        ids.view(-1)[:10]=torch.tensor([0,0,1,1,63,64,-1,v.min,v.max,32],dtype=dtype)
        offsets.view(-1)[:4]=torch.tensor([torch.iinfo(torch.int64).max-10,
                                           torch.iinfo(torch.int64).min+1,93,-177],dtype=torch.int64)
    return ids,offsets


def run(args,r):
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    gen=torch.Generator().manual_seed(1234)
    schedule=json.loads(args.schedule.read_bytes())
    assert len(schedule)==len(set(map(tuple,schedule)))==60
    stream0=torch.npu.current_stream()
    stream1=torch.npu.Stream()
    r.update(schedule=schedule, cases=[],versions={"torch":torch.__version__,"torch_npu":torch_npu.__version__})
    ok=True
    for dtype in (torch.int64,torch.int32):
        for i,(b,l) in enumerate(schedule):
            host,base=inputs(b,l,dtype,gen)
            cpu=impl.reference(host,base,64)
            ids,off=host.npu(),base.npu()
            stream=stream0 if i%2==0 else stream1
            stream.wait_stream(stream0)
            row={"shape":[b,l],"dtype":str(dtype),"stream":"default" if i%2==0 else "alternate","routes":{}}
            with torch.npu.stream(stream):
                for route in ("A","S","P"):
                    counts=impl.COUNTS.copy()
                    out=impl.prepare_ids(ids,off,64,route)
                    stream.synchronize()
                    delta=diff(out,cpu)
                    row["routes"][route]={"diff":delta,"counts":dict(impl.COUNTS-counts)}
                    ok &= not delta
            r["cases"].append(row)
            if not ok:
                r.update(status="integer_semantics_mismatch",plans=impl.INFO);return
    r["verified_shapes"]=len(schedule);r["initial_counts"]=dict(impl.COUNTS);r["plans"]=impl.INFO
    if impl.COUNTS["cpp_returned"]!=118:
        r["status"]="planned_route_not_active";return
    # Invalid layout / unsupported table population use the unmodified ATen path.
    a,o=inputs(2,14,torch.int64,gen,False); x=a.npu(); off=o.npu()
    variations=[("strided_ids",x.transpose(0,1),off,64),("rows",x,off,63),
                ("tables",x[:,:,:512].contiguous(),off[:512],64),
                ("strided_offsets",x,torch.stack([off,off],dim=1)[:,0],64)]
    r["fallbacks"]=[]
    for tag,v,t,n in variations:
        counts=impl.COUNTS.copy()
        actual=impl.prepare_ids(v,t,n,"P")
        delta=diff(actual,impl.reference(v,t,n))
        routes=dict(impl.COUNTS-counts)
        r["fallbacks"].append({"case":tag,"diff":delta,"routes":routes})
        if delta or not any(k.startswith("fallback_") for k in routes):
            r["status"]="fallback_check_failed";return
    # Allocate on default, consume on another stream, discard all input references before synchronization.
    # The allocator may immediately reuse a freed default-stream block.
    cpu,o=inputs(2,22,torch.int64,gen)
    expected=impl.reference(cpu,o,64)
    x,off=cpu.npu(),o.npu()
    stream1.wait_stream(stream0)
    with torch.npu.stream(stream1):
        actual=impl.prepare_ids(x,off,64,"P")
    del x,off
    junk=[torch.full(cpu.shape,i,dtype=cpu.dtype,device="npu") for i in range(8)]
    stream1.synchronize()
    r["cross_stream_release"]=diff(actual,expected)
    del junk
    if r["cross_stream_release"]:
        r["status"]="stream_lifetime_failed";return
    sets=[]
    for b,l in SHAPES:
        x,o=inputs(b,l,torch.int64,gen,False)
        sets.append((x.npu(),o.npu(),torch.empty((1024,b,l,4),dtype=torch.int64,device="npu")))
    for x,o,_ in sets:
        for route in "ASP": impl.prepare_ids(x,o,64,route)
    torch.npu.synchronize()
    r["windows"]=[]
    for mode,orders in (("including_allocation",("ASP","PSA","ASP","PSA")),("preallocated_same_kernel",("SP","PS","SP","PS"))):
        for order in orders:
            pair={}
            for route in order:
                counts=impl.COUNTS.copy(); host=[]
                torch.npu.synchronize(); start=time.perf_counter_ns()
                for i in range(args.repeats):
                    x,o,z=sets[i%len(sets)]
                    begin=time.perf_counter_ns()
                    out=impl.prepare_ids(x,o,64,route,z if mode=="preallocated_same_kernel" else None)
                    host.append((time.perf_counter_ns()-begin)/1000)
                torch.npu.synchronize()
                elapsed=(time.perf_counter_ns()-start)/1000
                pair[route]={"wall_us":elapsed,"per_call_us":elapsed/args.repeats,"calls":args.repeats,
                             "host_us":host,"p50_host_us":statistics.median(host),
                             "p95_host_us":sorted(host)[(len(host)-1)*95//100],"counts":dict(impl.COUNTS-counts)}
            r["windows"].append({"mode":mode,"order":order,"legs":pair})
    r["profile"]=[]
    if args.profile:
        p=torch_npu.profiler
        with p.profile(activities=[p.ProfilerActivity.CPU,p.ProfilerActivity.NPU],
                       on_trace_ready=p.tensorboard_trace_handler(str(args.profile)),
                       experimental_config=p._ExperimentalConfig(profiler_level=p.ProfilerLevel.Level1)):
            for i,shape in enumerate(SHAPES):
                x,o,_=sets[i]
                for rep,order in enumerate(("ASP","PSA","ASP")):
                    for route in order:
                        counts=impl.COUNTS.copy();torch.npu.synchronize()
                        tag=f"SideIDs/{shape[0]}x{shape[1]}/{rep}/{route}"
                        with torch.autograd.profiler.record_function(tag):
                            out=impl.prepare_ids(x,o,64,route)
                            torch.npu.synchronize()
                        r["profile"].append({"tag":tag,"counts":dict(impl.COUNTS-counts)})
    r["status"]="passed";r["counts"]=dict(impl.COUNTS)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("schedule","out"):cli.add_argument("--"+n,type=Path,required=True)
    cli.add_argument("--profile",type=Path)
    cli.add_argument("--repeats",type=int,default=48)
    a=cli.parse_args(); rep={}
    try:run(a,rep)
    finally:a.out.write_text(json.dumps(rep,indent=2)+"\n")
    sys.exit(0 if rep.get("status")=="passed" else 2)
