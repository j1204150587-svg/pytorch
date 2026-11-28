"""Add replay admission, real owner/address/release checks, hot A/B + B/A and same-stream Level1."""
import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


FIELDS = ("accepted", "fallback", "new_plan", "rebind", "launch", "refused", "bind_failure", "other_stream", "single_use")


def dump(out, report):
    out.write_text(json.dumps(report, indent=2))


def run(a):
    import torch
    import torch_npu
    torch.npu.set_device(0)
    schedule=json.loads(a.schedule.read_bytes())
    owner=torch.npu.current_stream()
    lib=ctypes.CDLL(str(Path(torch_npu.__file__).parent/"lib/libtorch_npu.so"))
    get=lib.npu_repeat_add_count
    get.argtypes=[ctypes.c_uint]
    get.restype=ctypes.c_uint64
    toggle=lib.npu_repeat_add_profile_stock
    toggle.argtypes=[ctypes.c_bool]
    toggle.restype=None
    def counters():
        return {name:get(i) for i,name in enumerate(FIELDS)}
    gen=torch.Generator().manual_seed(89652)
    def pair(i, special=False):
        b,l=schedule[i]
        operands=[torch.randn(b,l,64,generator=gen).bfloat16() for _ in range(2)]
        if special:
            words=torch.tensor([0,-32768,1,-32767,127,-32641,128,-32640,32639,-129,
                                32640,-128,32705,-63,32641,-127],dtype=torch.int16)
            for j,x in enumerate(operands):
                x.view(torch.int16).view(-1)[:len(words)].copy_(words.roll(j))
        return tuple(x.npu() for x in operands)
    def ref(x,y,alpha=1):
        result=torch.empty(torch.broadcast_shapes(x.shape,y.shape),dtype=x.dtype,device=x.device)
        return torch.add(x,y,alpha=alpha,out=result)
    def same(x,y):
        return torch.equal(x.detach().cpu().contiguous().view(torch.int16),y.detach().cpu().contiguous().view(torch.int16))
    data={"mode":a.mode}
    try:
        if a.mode=="sem":
            tests=[]
            for i in range(len(schedule)):
                bank=[pair(i,i in (0,4,5,59)) for _ in range(3)]
                results=[torch.add(x,y) for x,y in bank]
                refs=[ref(x,y) for x,y in bank]
                addresses=[t.untyped_storage().data_ptr() for t in [*(z for xy in bank for z in xy),*results]]
                tests.append({"step":i,"shape":schedule[i],"distinct":len(set(addresses))==len(addresses),
                              "bits":all(same(z,r) for z,r in zip(results,refs))})
                if i==0:
                    data["first_counters"]=counters()
                    if get(2)!=1 or get(3)!=2 or get(4)!=3 or get(5):
                        data["cases"]=tests
                        raise RuntimeError("CANN repeatable admission, rebinding or guarded activation failed")
            data["cases"]=tests
            x,y=pair(4)
            special=[(x,y,2),(x,y[:,:,:1],1),(x.transpose(0,1),y.transpose(0,1),1)]
            torch.npu.synchronize()
            before=counters()
            data["fallback_bits"]=[same(torch.add(u,v,alpha=alpha),ref(u,v,alpha)) for u,v,alpha in special]
            data["fallback_counts"]={k:v-before[k] for k,v in counters().items()}
            other=torch.npu.Stream()
            jobs=[(s,*pair(j)) for s,j in ((owner,4),(other,4),(owner,5),(other,5))]
            refs=[ref(x,y) for _,x,y in jobs]
            torch.npu.synchronize()
            def launch(job):
                stream,x,y=job
                torch.npu.set_device(0)
                with torch.npu.stream(stream):
                    return [torch.add(x,y) for _ in range(12)]
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
                out=list(pool.map(launch,jobs))
            torch.npu.synchronize()
            data["threaded_owner_bits"]=all(same(v,r) for vs,r in zip(out,refs) for v in vs)
            live=[]
            for j in range(42):
                x,y=pair(4+j%2)
                rr=ref(x,y)
                z=torch.add(x,y)
                del x,y
                live.append((z,rr))
            torch.npu.synchronize()
            data["released_inputs_bits"]=all(same(z,rr) for z,rr in live)
            c=counters()
            data["passed"]=(all(t["distinct"] and t["bits"] for t in tests) and all(data["fallback_bits"])
                            and data["fallback_counts"]["fallback"]>=3 and data["threaded_owner_bits"]
                            and data["released_inputs_bits"] and c["new_plan"]==len(set(map(tuple,schedule)))
                            and c["other_stream"]>=24 and c["accepted"]==c["new_plan"]+c["rebind"]==c["launch"]
                            and c["refused"]==c["bind_failure"]==c["single_use"]==0)
        elif a.mode=="fault":
            bits=[]
            for j in range(6):
                x,y=pair(4)
                bits.append(same(torch.add(x,y),ref(x,y)))
            c=counters()
            data["bits"]=bits
            data["passed"]=(all(bits) and c["bind_failure"]==c["single_use"]==1 and c["launch"]==2 and c["fallback"]==4)
        else:
            bank=[pair(i) for i in range(len(schedule))]
            def window(rows,iterations):
                torch.npu.synchronize()
                start=time.perf_counter_ns()
                for _ in range(iterations):
                    for x,y in rows:
                        z=torch.add(x,y)
                torch.npu.synchronize()
                return (time.perf_counter_ns()-start)/(1000*len(rows)*iterations)
            if a.mode=="profile":
                for stock in (True,False):
                    toggle(stock)
                    window(bank,6)
                prof=torch_npu.profiler
                with prof.profile(activities=[prof.ProfilerActivity.CPU,prof.ProfilerActivity.NPU],record_shapes=True,
                                  experimental_config=prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1),
                                  on_trace_ready=prof.tensorboard_trace_handler(str(a.output.parent/"profile"))):
                    for step,order in [(4,"AB"),(5,"BA")]:
                        for leg in order:
                            toggle(leg=="A")
                            with torch.profiler.record_function(f"RepeatAdd_{leg}_{step}"):
                                window(bank[step:step+1],8)
                data["passed"]=True
            else:
                window(bank,10)
                data["us_per_op"]={k:[window(rows,n) for _ in range(9)] for k,rows,n in
                                   (("shape4",bank[4:5],512),("shape5",bank[5:6],512),("all60",bank,16))}
                data["passed"]=True
    except BaseException as ex:
        data.update(error=repr(ex),passed=False)
        raise
    finally:
        data["counters"]=counters()
        dump(a.output,data)
    if not data["passed"]:
        raise SystemExit(2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--schedule",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--mode")
    a=p.parse_args()
    if a.mode:
        run(a)
        return
    data={"shape_sha256":hashlib.sha256(a.schedule.read_bytes()).hexdigest(),"cases":[]}
    for i,m in enumerate(("sem","fault","A","B","B","A","profile")):
        env=dict(os.environ,ACLNN_CACHE_LIMIT="65536",NPU_DYNAMIC_REBASE_INDICES="1",NPU_DYNAMIC_REBASE_FORWARD="1",
                 NPU_HSTU_REPEAT_ADD=str(int(m!="A")),NPU_HSTU_REPEAT_ADD_COUNT=str(int(m in ("sem","fault","profile"))),
                 NPU_HSTU_REPEAT_ADD_PROFILE_TOGGLE=str(int(m=="profile")),
                 NPU_HSTU_REPEAT_ADD_TEST_REBIND_FAILURE=str(int(m=="fault")))
        out=a.output.with_name(f"part-{i}-{m}.json")
        rc=subprocess.run([sys.executable,"-B",__file__,"--schedule",str(a.schedule),"--output",str(out),"--mode",m],env=env).returncode
        r=json.loads(out.read_bytes()) if out.exists() else {}
        data["cases"].append({"mode":m,"rc":rc,"file":out.name,"counters":r.get("counters"),
                              "median_us":{k:statistics.median(v) for k,v in r.get("us_per_op",{}).items()}})
        dump(a.output,data)
        if rc:
            raise SystemExit(rc)
    data["complete"]=True
    dump(a.output,data)


if __name__=="__main__":
    main()
