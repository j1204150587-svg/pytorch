"""Fail-fast ACLNN repeatability and same-stream output/address ownership probe."""
import argparse
import ctypes as C
import json
from pathlib import Path
from time import perf_counter_ns

import torch
import torch_npu


def load(path):
    lib=C.CDLL(str(path.resolve()))
    ptr=C.c_void_p
    i64p=C.POINTER(C.c_int64)
    for name,args,result in (
        ("new",[],ptr),("tensor",[ptr,C.c_int,i64p,C.c_int,i64p,C.c_int64,C.c_int64,ptr],C.c_int),
        ("prepare",[ptr,C.POINTER(C.c_uint64)],C.c_int),("repeat",[ptr],C.c_int),
        ("rebind",[ptr,C.c_int,ptr],C.c_int),("launch",[ptr,ptr,ptr],C.c_int),("close",[ptr],C.c_int)):
        f=getattr(lib,"probe_"+name)
        f.argtypes=args
        f.restype=result
    return lib


def metadata(t,normalize):
    item=t.element_size()
    storage=t.untyped_storage()
    extent=storage.nbytes()//item
    offset=t.storage_offset()
    start=0
    if normalize:
        quantum=512//item
        start=offset//quantum*quantum
        offset-=start
        need=offset+1+sum((n-1)*s for n,s in zip(t.shape,t.stride(),strict=True))
        bounded=(need+quantum-1)//quantum*quantum
        if start+bounded>extent:
            raise ValueError("rebased descriptor would exceed owned storage")
        extent=bounded
    base=storage.data_ptr()+start*item
    if base%32:
        raise ValueError("CANN requires at least 32-byte base alignment")
    return (tuple(t.shape),tuple(t.stride()),offset,extent),base


def make_args(batch,length,geometry):
    cases=[]
    # Each case owns separate ID, grad and output storage, held through the final event.
    # Table numbers differ by 16, preserving the 512-byte residual within a normalized family.
    for i,t in enumerate((1,17,33,49)):
        table=1 if geometry=="exact" else t
        source=torch.arange(batch*length*1024*4,dtype=torch.int64)
        bank=((source*7+source//5+3*i)%29).view(batch,length,1024,4)
        ids=bank.npu()[:,:,table,:]
        grad=((torch.arange(batch*length*4*64).view(batch,length,4,64)+i*3)%9-4).to(torch.bfloat16).npu()
        out=torch.full((64,64),511.0,dtype=torch.bfloat16,device=grad.device)
        expected=torch.ops.aten.embedding_dense_backward.default(grad,ids,64,-1,False).cpu()
        cpu_ref=torch.ops.aten.embedding_dense_backward.default(grad.cpu(),bank[:,:,table,:],64,-1,False)
        if not torch.equal(expected,cpu_ref):
            raise ValueError("the stock oracle does not agree with the exact integer-gradient reference")
        cases.append((grad,ids,out,expected))
    torch.npu.synchronize()
    return cases


class ProbeFailure(RuntimeError):
    pass


def experiment(lib,cases,geometry,repeat,pipelined,root,persist):
    status=[]
    root["events"]=status
    held=[]
    handles=[]
    shape_key=None
    ws=None
    handle=None
    stream=torch.npu.current_stream()
    def call(name,func,*args):
        t=perf_counter_ns()
        rc=func(*args)
        status.append(dict(name=name,rc=rc,host_us=(perf_counter_ns()-t)/1000))
        if rc:
            root["failure"]=dict(name=name,rc=rc)
            persist()
            raise ProbeFailure(f"{name} returned {rc}")
    for index,(grad,ids,out,expected) in enumerate(cases):
        meta=[metadata(t,geometry=="bounded" and slot==1) for slot,t in enumerate((grad,ids,out))]
        keys=[m[0] for m in meta]
        if index==0:
            root["descriptor_shapes_strides_offset_extent"]=keys
            shape_key=keys
        if repeat and keys!=shape_key:
            raise ValueError("attempted to change metadata using only an address setter")
        if len({b for _,b in meta})!=3:
            raise ValueError("aliased input/output storage in probe")
        if not repeat or index==0:
            handle=lib.probe_new()
            if not handle:
                raise MemoryError("probe_new")
            handles.append(handle)
            for slot,((dims,strides,offset,extent),base) in enumerate(meta):
                rank=len(dims)
                shape=(C.c_int64*rank)(*dims)
                stride=(C.c_int64*rank)(*strides)
                call(f"create_{slot}",lib.probe_tensor,handle,slot,shape,rank,stride,offset,extent,base)
            n=C.c_uint64()
            call("GetWorkspaceSize",lib.probe_prepare,handle,C.byref(n))
            root.setdefault("workspace_bytes",[]).append(n.value)
            # Same host thread calls this immediately after a successful stage 1,
            # before allocating or issuing any other ACLNN operation.
            if repeat:
                try:
                    call("set_repeatable",lib.probe_repeat,handle)
                except ProbeFailure:
                    # It is still a single-use executor. Issue a normal stage 2
                    # before close so the documented ownership contract applies.
                    tmp=torch.empty(max(1,n.value),dtype=torch.uint8,device=grad.device)
                    call("single_use_cleanup_launch",lib.probe_launch,handle,tmp.data_ptr(),stream.npu_stream)
                    stream.synchronize()
                    call("single_use_cleanup_close",lib.probe_close,handle)
                    raise
            ws=torch.empty(max(1,n.value),dtype=torch.uint8,device=grad.device)
            held.append(ws)
        else:
            for slot,(_,base) in enumerate(meta):
                call(f"rebind_{slot}",lib.probe_rebind,handle,slot,base)
        with torch.profiler.record_function(f"RepeatProbe/{geometry}/{'B' if repeat else 'A'}/launch{index}"):
            call("stage2",lib.probe_launch,handle,ws.data_ptr(),stream.npu_stream)
        if not pipelined:
            stream.synchronize()
            if not torch.equal(out.cpu(),expected):
                root["failure"]=dict(name="early_output_mismatch",call=index)
                persist()
                raise ProbeFailure(f"incorrect result on launch {index}")
    stream.synchronize()
    same=[torch.equal(out.cpu(),expected) for _,_,out,expected in cases]
    root["independent_outputs_equal"]=same
    for p in handles:
        call("close_after_stream_sync",lib.probe_close,p)
    root["passed"]=all(same)
    persist()
    if not all(same):
        raise ProbeFailure("wrong output or older owner overwritten")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--geometry",choices=("exact","raw-family","bounded"),required=True)
    parser.add_argument("--mode",choices=("stock","repeat"),required=True)
    parser.add_argument("--pipelined",action="store_true",help="issue four requests on one stream before synchronizing")
    parser.add_argument("--profile",type=Path,help="optional torch_npu Level1 output directory")
    args=parser.parse_args()
    if args.geometry=="raw-family" and args.mode=="repeat":
        raise ValueError("raw-family changes storage offset across tables; use as a stock-only reference")
    torch.npu.set_device(0)
    lib=load(args.library)
    report=dict(torch=torch.__version__,torch_npu=torch_npu.__version__,geometry=args.geometry,
                mode=args.mode,pipelined=args.pipelined,library=str(args.library),runs=[])
    def persist():
        args.output.write_text(json.dumps(report,indent=2))
    try:
        # Prepare framework oracles and inputs before capturing submitted tasks.
        groups=[(shape,make_args(*shape,args.geometry)) for shape in ((1,19),(2,14))]
        from contextlib import nullcontext
        if args.profile:
            import torch_npu.profiler as np
            profiler=np.profile(activities=[np.ProfilerActivity.CPU,np.ProfilerActivity.NPU],
                on_trace_ready=np.tensorboard_trace_handler(str(args.profile)),
                experimental_config=np._ExperimentalConfig(profiler_level=np.ProfilerLevel.Level1))
        else:
            profiler=nullcontext()
        with profiler:
            for shape,cases in groups:
                run=dict(shape=shape)
                report["runs"].append(run)
                persist()
                experiment(lib,cases,args.geometry,args.mode=="repeat",args.pipelined,run,persist)
        report["passed"]=True
    except Exception as exc:
        report.update(passed=False,error=str(exc))
        # Pending raw launches may reference resources visible only in the
        # failing frame. Drain the stream before unwinding its owners.
        try:
            torch.npu.current_stream().synchronize()
        except Exception as sync_error:
            report["drain_error"]=str(sync_error)
        raise
    finally:
        persist()


if __name__=="__main__":
    main()
