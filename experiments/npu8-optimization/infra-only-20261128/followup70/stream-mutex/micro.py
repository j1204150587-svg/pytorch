"""Verify real Stream creation/destruction and measure hot aclnnAdd on 60 dynamic sizes."""
import argparse
import ctypes as ct
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE=Path(__file__).resolve().parent
PRIVATE=HERE/"private/libruntime_v100.so"
PROBE=HERE.parent/"cann-pending/launch_probe.so"
V=ct.c_void_p
U=ct.c_uint64


def check(rc):
    if rc: raise RuntimeError(f"ACL returned {rc}")


def run(arm,schedule,out,profile):
    import torch
    import torch_npu
    torch.npu.set_device(0)
    current=torch.npu.current_stream()
    assert (torch.tensor([4.0],device="npu")+4).cpu().item()==8
    libs={Path(row.split(maxsplit=5)[-1]).resolve() for row in Path("/proc/self/maps").read_text().splitlines()
          if row.endswith("/libruntime_v100.so")}
    assert libs=={PRIVATE.resolve()},libs
    native=ct.CDLL(str(PRIVATE),mode=os.RTLD_NOLOAD|os.RTLD_NOW)
    count=native.npu_cann_stream_layout_count
    count.argtypes=[ct.c_uint]
    count.restype=U
    classify=native.npu_cann_stream_layout_classify
    classify.argtypes=[U]
    classify.restype=ct.c_uint
    object_of=native.npu_cann_stream_layout_object
    object_of.argtypes=[U]
    object_of.restype=U
    rt=ct.CDLL("libascendcl.so")
    create=rt.aclrtCreateStream
    create.argtypes=[ct.POINTER(V)]
    destroy=rt.aclrtDestroyStream
    destroy.argtypes=[V]
    initial=[count(i) for i in range(5)]
    created=[]
    for i in range(4):
        stream=V()
        check(create(ct.byref(stream)))
        created.append((stream.value,classify(stream.value)))
    for addr,_ in reversed(created): check(destroy(V(addr)))
    after=[count(i) for i in range(5)]
    assert after[0]-initial[0]>=4
    if arm=="A": assert after[2]==after[3]==after[4]==0
    else: assert after[2]==after[0] and after[1]+after[4]==after[0]
    # Bind both legs to the same fresh ACL stream creation ordinal. Torch may
    # also expose a default-stream sentinel; retain its handle separately.
    measured=V()
    check(create(ct.byref(measured)))
    stream=measured.value
    cl=classify(stream)
    shapes=json.loads(schedule.read_bytes())
    assert len(set(map(tuple,shapes)))==len(shapes)==60
    data=dict(arm=arm,library_sha256=hashlib.sha256(PRIVATE.read_bytes()).hexdigest(),
              stream=stream,framework_default=current.npu_stream,stream_object=int(object_of(stream)),
              current_geometry=cl,created=created,counters_before=initial,counters_after=after,
              shape_sequence=shapes,probe_sha256=hashlib.sha256(PROBE.read_bytes()).hexdigest())
    out.write_text(json.dumps(data))
    assert all(g in (1,2) for _,g in created),data
    assert cl in (1,2),("current stream outside factory",data)
    probe=ct.CDLL(str(PROBE))
    prepare=probe.pending_probe_prepare
    prepare.argtypes=[ct.c_int64,ct.c_int64,V,V,V,ct.POINTER(V)]
    submit=probe.pending_probe_window
    submit.argtypes=[ct.POINTER(V),ct.c_uint,ct.c_uint,V,ct.POINTER(U),ct.POINTER(U),ct.POINTER(U)]
    close=probe.pending_probe_close
    close.argtypes=[V]
    current_cpu=ct.CDLL(None).sched_getcpu
    current_cpu.argtypes=[]
    current_cpu.restype=ct.c_int
    refs=[]
    plans=[]
    def window(ids,n,individual=False):
        ps=(V*len(ids))(*(plans[i] for i in ids))
        host,full=U(),U()
        lat=(U*n)() if individual else None
        start_cpu=current_cpu()
        check(submit(ps,len(ps),n,stream,lat,ct.byref(host),ct.byref(full)))
        return dict(n=n,host_ns=host.value,complete_ns=full.value,cpu_before=start_cpu,cpu_after=current_cpu(),
                    **({"launch_ns":list(lat)} if lat is not None else {}))
    try:
        generator=torch.Generator().manual_seed(221486)
        for b,l in shapes:
            x,y=(torch.randn(b,l,64,generator=generator).bfloat16() for _ in range(2))
            nx,ny=x.npu(),y.npu()
            result=torch.empty_like(nx)
            plan=V()
            check(prepare(b,l,nx.data_ptr(),ny.data_ptr(),result.data_ptr(),ct.byref(plan)))
            refs.append((nx,ny,result,x+y))
            plans.append(plan.value)
        all_ids=list(range(60))
        # Source tensors are filled on Torch's stream. All uses on the separate
        # measured stream follow this setup synchronization, outside its windows.
        torch.npu.synchronize()
        window(all_ids,180)
        data["bitwise"]=[torch.equal(z.cpu().view(torch.int16),ref.view(torch.int16)) for _,_,z,ref in refs]
        assert all(data["bitwise"])
        data["windows"]={}
        if arm=="B" and cl!=2: data["current_fallback"]=True
        for key,ids,n in (("first-shape4",[4],512),("first-shape5",[5],512),("all60",all_ids,480)):
            window(ids,n)
            unprofiled=[window(ids,n) for _ in range(21)]
            individual=[window(ids,n,True) for _ in range(3)]
            data["windows"][key]=dict(unprofiled=unprofiled,individual=individual)
        if profile:
            prof=torch_npu.profiler
            profile_dir=out.parent/(out.stem+"-profile")
            with prof.profile(activities=[prof.ProfilerActivity.CPU,prof.ProfilerActivity.NPU],record_shapes=True,
                    experimental_config=prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1),
                    on_trace_ready=prof.tensorboard_trace_handler(str(profile_dir))):
                for j in (4,5):
                    with torch.profiler.record_function(f"StreamLayout_{arm}_{j}"):
                        window([j],32)
        data["complete"]=True
    finally:
        torch.npu.synchronize()
        for handle in plans: check(close(handle))
        check(destroy(measured))
        out.write_text(json.dumps(data))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out",type=Path)
    parser.add_argument("schedule",type=Path)
    parser.add_argument("--arm",choices="AB")
    parser.add_argument("--profile",action="store_true")
    args=parser.parse_args()
    if args.arm:
        run(args.arm,args.schedule,args.out,args.profile)
        return
    cases=[]
    for i,arm in enumerate("ABBA"):
        result=args.out.parent/f"part-{i}-{arm}.json"
        env=dict(os.environ,NPU_CANN_STREAM_LAYOUT=str(int(arm=="B")),
                 LD_LIBRARY_PATH=str(PRIVATE.parent)+":"+os.environ.get("LD_LIBRARY_PATH",""))
        sub=subprocess.run([sys.executable,"-B",__file__,str(result),str(args.schedule),"--arm",arm,
                            *(["--profile"] if i<2 else [])],env=env)
        cases.append(dict(arm=arm,rc=sub.returncode,result=result.name))
        args.out.write_text(json.dumps(cases,indent=2))
        if sub.returncode: raise SystemExit(sub.returncode)


if __name__=="__main__": main()
