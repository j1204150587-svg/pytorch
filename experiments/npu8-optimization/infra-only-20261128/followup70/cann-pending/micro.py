"""Run hot CANN Add stage 2 with real launches, address-stable 60-shape plans and one native runtime variable."""
import argparse
import ctypes as C
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

HERE=Path(__file__).resolve().parent
PRIVATE=HERE/"source-v2/libruntime_v100.so"
STOCK=Path("/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libruntime_v100.so")
U=C.c_uint64
V=C.c_void_p


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(a, report):
    import torch
    import torch_npu
    torch.npu.set_device(0)
    gen=torch.Generator().manual_seed(221486)
    libc=C.CDLL(None)
    getcpu=libc.sched_getcpu
    getcpu.argtypes=[]
    getcpu.restype=C.c_int
    report["cpu_affinity"]=sorted(os.sched_getaffinity(0))
    sg=json.loads(a.schedule.read_bytes())
    st=torch.npu.current_stream()
    x0=torch.tensor([4.0],dtype=torch.bfloat16,device="npu")
    if (x0+x0).cpu().item()!=8.:raise ValueError("First real NPU Add")
    paths={Path(line.split(maxsplit=5)[-1].strip()).resolve() for line in Path("/proc/self/maps").read_text().splitlines()
           if line.rstrip().endswith("/libruntime_v100.so")}
    expected=STOCK if a.mode=="S" else PRIVATE
    report["mapped"]=[str(p) for p in paths]
    if paths!={expected.resolve()}:raise ValueError(("wrong runtime mappings",report["mapped"]))
    report.update(mapped_sha=digest(expected),framework_current_stream=int(st.npu_stream))
    rlib=C.CDLL(str(expected),mode=os.RTLD_NOLOAD|os.RTLD_NOW)
    count=toggle=None
    if a.mode!="S":
        count=rlib.npu_cann_pending_count
        count.argtypes=[C.c_uint]
        count.restype=U
        toggle=rlib.npu_cann_pending_profile_select
        toggle.argtypes=[C.c_bool]
        toggle.restype=None
        report.update(hwcap=bool(count(2)),active=int(count(3)))
        if not count(2) or count(3)!=int(a.mode in ("B","gateB")):raise ValueError(report)
    lib=C.CDLL(str(HERE/"launch_probe.so"))
    make=lib.pending_probe_prepare
    make.argtypes=[C.c_int64,C.c_int64,V,V,V,C.POINTER(V)]
    submit=lib.pending_probe_window
    submit.argtypes=[C.POINTER(V),C.c_uint,C.c_uint,V,C.POINTER(U),C.POINTER(U),C.POINTER(U)]
    close=lib.pending_probe_close
    close.argtypes=[V]
    def check(rc):
        if rc:raise RuntimeError(f"ACL return code {rc}")
    handles=[]
    owners=[]
    def window(which,n,per=False):
        source=(V*len(which))(*(handles[i] for i in which))
        host,full=U(),U()
        latency=(U*n)() if per else None
        start_cpu=getcpu()
        check(submit(source,len(source),n,st.npu_stream,latency,C.byref(host),C.byref(full)))
        return dict(n=n,host_ns=host.value,complete_ns=full.value,cpu_before=start_cpu,cpu_after=getcpu(),
                    **({"launch_ns":list(latency)} if per else {}))
    try:
        for b,l in sg:
            x,y=(torch.randn(b,l,64,generator=gen).bfloat16() for _ in range(2))
            nx,ny=x.npu(),y.npu()
            result=torch.empty_like(nx)
            ref=x+y
            holder=V()
            check(make(b,l,nx.data_ptr(),ny.data_ptr(),result.data_ptr(),C.byref(holder)))
            handles.append(holder.value)
            owners.append((nx,ny,result,ref))
        torch.npu.synchronize()
        all_shapes=list(range(len(sg)))
        window(all_shapes,len(sg)*3)
        if a.mode in ("S","gateA","gateB"):
            before=[int(count(i)) for i in (0,1)] if count else []
            window(all_shapes,len(sg)*4)
            after=[int(count(i)) for i in (0,1)] if count else []
            report["counter_delta"]=[b-a for a,b in zip(before,after)]
            report["expected_add_launches"]=len(sg)*4
            report["bitwise_shapes"]=[torch.equal(z.cpu().view(torch.int16),ref.view(torch.int16)) for _,_,z,ref in owners]
            if not all(report["bitwise_shapes"]):raise ValueError("Unexpected output")
            if count:
                chosen=int(a.mode=="gateB")
                # Two stream synchronizations may have control commands beyond the 240 Add work items.
                if report["counter_delta"][chosen]<len(sg)*4 or report["counter_delta"][1-chosen]:
                    raise ValueError(("native activation did not cover the selected real launches",report["counter_delta"]))
            if a.mode=="S":
                gc.disable()
                report["stock_reference"]={}
                for key,idx,n in (("shape4",[4],512),("shape5",[5],512),("all60",all_shapes,480)):
                    window(idx,n)
                    report["stock_reference"][key]=[window(idx,n) for _ in range(21)]
        elif a.mode=="paired":
            gc.disable()
            report["paired"]={}
            for k,idx,n in (("shape4",[4],512),("shape5",[5],512),("all60",all_shapes,480)):
                for arm in "AB":
                    toggle(arm=="B")
                    window(idx,n)
                sets=[]
                for order in ["AB","BA"]*12:
                    row={"order":order}
                    for arm in order:
                        toggle(arm=="B")
                        if count(3)!=int(arm=="B"):raise ValueError(("unexpected path",arm))
                        row[arm]=window(idx,n)
                    sets.append(row)
                report["paired"][k]=sets
        elif a.mode=="profile":
            prof=torch_npu.profiler
            with prof.profile(activities=[prof.ProfilerActivity.CPU,prof.ProfilerActivity.NPU],record_shapes=True,
                              experimental_config=prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1),
                              on_trace_ready=prof.tensorboard_trace_handler(str(a.output.parent/"profile"))):
                for j,order in ((4,"AB"),(5,"BA")):
                    for arm in order:
                        toggle(arm=="B")
                        before=[int(count(i)) for i in (0,1)]
                        with torch.profiler.record_function(f"Pending_{arm}_{j}"):
                            window([j],32)
                        delta=[count(i)-before[i] for i in (0,1)]
                        chosen=int(arm=="B")
                        report.setdefault("profile_arms",[]).append(dict(shape=sg[j],arm=arm,counter_delta=delta))
                        if delta[chosen]<32 or delta[1-chosen]:raise ValueError(delta)
        else:
            gc.disable()
            report["windows"]={}
            for k,idx,n in (("shape4",[4],512),("shape5",[5],512),("all60",all_shapes,480)):
                window(idx,n)
                windows=[window(idx,n) for _ in range(21)]
                micros=[window(idx,n,True) for _ in range(4)]
                report["windows"][k]=dict(no_call_timing=windows,individual=micros)
        report["passed"]=True
    finally:
        torch.npu.synchronize()
        for handle in handles:check(close(handle))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--schedule",required=True,type=Path)
    cli.add_argument("--output",required=True,type=Path)
    cli.add_argument("--mode")
    a=cli.parse_args()
    if a.mode:
        report={"mode":a.mode}
        try:run(a,report)
        except BaseException as exc:
            report.update(passed=False,error=repr(exc))
            raise
        finally:write(a.output,report)
        return
    report={"source":digest(a.schedule),"cases":[]}
    # Only private processes get the replacement search path. The toolkit and container installation remain shared.
    for i,mode in enumerate(("S","gateA","gateB","A","B","B","A","paired","profile")):
        env=dict(os.environ,NPU_CANN_PENDING_LSE=str(int(mode in ("B","gateB"))),
                 NPU_CANN_PENDING_COUNT=str(int(mode in ("gateA","gateB","profile"))),
                 NPU_CANN_PENDING_PROFILE=str(int(mode in ("paired","profile"))),ACLNN_CACHE_LIMIT="65536",
                 NPU_DYNAMIC_REBASE_INDICES="1",NPU_DYNAMIC_REBASE_FORWARD="1")
        if mode!="S":env["LD_LIBRARY_PATH"]=str(PRIVATE.parent)+":"+env.get("LD_LIBRARY_PATH","")
        out=a.output.with_name(f"part-{i}-{mode}.json")
        rc=subprocess.run([sys.executable,"-B",__file__,"--mode",mode,"--schedule",str(a.schedule),"--output",str(out)],env=env).returncode
        r=json.loads(out.read_bytes()) if out.exists() else {}
        report["cases"].append(dict(mode=mode,rc=rc,passed=r.get("passed",False),error=r.get("error"),mapped=r.get("mapped"),
            counter_delta=r.get("counter_delta"),median_host_us={k:statistics.median(x["host_ns"]/x["n"]/1000 for x in d["no_call_timing"])
            for k,d in r.get("windows",{}).items()}))
        write(a.output,report)
        if rc:raise SystemExit(rc)
    report["complete"]=True
    write(a.output,report)


if __name__=="__main__":main()
