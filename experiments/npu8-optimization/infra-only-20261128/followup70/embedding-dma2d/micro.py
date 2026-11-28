"""Check CANN 9.1 two-dimensional D2D support for one side-table index view."""
import ctypes as c
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
import torch_npu


schedule=json.loads(Path(sys.argv[1]).read_bytes())
dest=Path(sys.argv[2])
torch.npu.set_device(0)
lib=c.CDLL("libascendcl.so")
p,size=c.c_void_p,c.c_size_t
copy2=lib.aclrtMemcpy2dAsync
copy2.argtypes=[p,size,p,size,size,size,c.c_int,p]
copy1=lib.aclrtMemcpyAsync
copy1.argtypes=[p,size,p,size,c.c_int,p]
sync=lib.aclrtSynchronizeStream
sync.argtypes=[p]
report=dict(soc=torch.npu.get_device_name(0),results=[],profiles=[],complete=False)
streams=[torch.npu.current_stream(),torch.npu.Stream()]


def point(k, shape, table, stream, profiled=False):
    b,length=shape
    source_bytes=b*length*32768
    data=torch.arange(source_bytes//8,dtype=torch.int64).reshape(b,length,1024,4)+17+k
    ref=data[:,:,table,:].contiguous()
    dev=data.npu()
    first=dev[:1,:1,table,:].reshape(4)
    storage=dev.untyped_storage()
    assert source_bytes==storage.nbytes()
    offset=table*32
    assert offset+(b*length-1)*32768+32<=storage.nbytes()
    assert first.data_ptr()==storage.data_ptr()+offset
    torch.npu.synchronize()
    for leg in ("ref","D2D1","D2D2"):
        width,height,pitch=32,b*length,32768
        target=torch.full((b,length,4),-777,dtype=torch.int64,device=dev.device)
        torch.npu.synchronize()
        rc=0
        need=target.numel()*target.element_size()
        assert need==height*width and need==target.untyped_storage().nbytes()
        name=f"studyDMA_{leg}_{k}_{table}"
        with torch.profiler.record_function(name):
            launch_start=time.perf_counter_ns()
            if leg=="ref":
                result=dev[:,:,table,:].contiguous()
            elif leg=="D2D1":
                rc=copy1(target.data_ptr(),need,first.data_ptr(),width,3,stream.npu_stream)
                result=target
            else:
                rc=copy2(target.data_ptr(),width,first.data_ptr(),pitch,width,height,3,stream.npu_stream)
                result=target
            return_time=time.perf_counter_ns()
            sync_rc=sync(stream.npu_stream)
            sync_time=time.perf_counter_ns()
        host=result.cpu()
        if leg=="D2D1":
            expected=torch.full_like(ref,-777)
            expected.reshape(-1)[:4]=ref.reshape(-1)[:4]
        else:
            expected=ref
        rec=dict(label=name,leg=leg,shape=shape,table=table,src_offset=offset,source_nbytes=source_bytes,
                 width=width,height=height,src_pitch=pitch,dst_pitch=width,kind=3,stream=stream.npu_stream,
                 rc=rc,sync_rc=sync_rc,host_call_ns=return_time-launch_start,sync_ns=sync_time-return_time,
                 ref_word_differences=int((host!=expected).sum()),
                 all_sentinel=bool((host==-777).all()),expected_sha256=hashlib.sha256(expected.numpy().tobytes()).hexdigest(),
                 actual_sha256=hashlib.sha256(host.numpy().tobytes()).hexdigest())
        report["profiles" if profiled else "results"].append(rec)
        dest.write_text(json.dumps(report,indent=2))
        assert sync_rc==0
        if leg!="D2D2":
            assert rc==0 and rec["ref_word_differences"]==0,rec
        elif rc==0:
            assert rec["ref_word_differences"]==0,rec
        else:
            assert rec["all_sentinel"],rec


for k,shape in enumerate(schedule):
    with torch.npu.stream(streams[k%2]):
        point(k,shape,(0,15,1023)[k%3],streams[k%2])
prof=torch_npu.profiler
with prof.profile(activities=[prof.ProfilerActivity.CPU,prof.ProfilerActivity.NPU],record_shapes=True,
                  experimental_config=prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1),
                  on_trace_ready=prof.tensorboard_trace_handler(str(dest.parent/"profile"))):
    for k,s in enumerate(([1,19],[2,14],[2,22])):
        with torch.npu.stream(streams[k%2]):
            point(k,s,(0,15,1023)[k],streams[k%2],True)
report["complete"]=True
dest.write_text(json.dumps(report,indent=2))

