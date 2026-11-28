"""Check scoped producer-to-consumer handoff against exact BF16 bit patterns and owner lifetimes."""
import argparse
import ctypes
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    p.add_argument("--switch",required=True,choices=("0","1"))
    cfg=p.parse_args()
    os.environ.update(NPU_INPLACE_COPY_CONSUMER=cfg.switch,NPU_DYNAMIC_REBASE_COUNT="1")
    import torch
    import torch_npu
    torch.npu.set_device(0)
    library,={r.split()[-1] for r in Path("/proc/self/maps").read_text().splitlines()
              if r.split()[-1].endswith("/libtorch_npu.so")}
    counter=ctypes.CDLL(library).npu_inplace_copy_rebase_counter
    counter.argtypes=(ctypes.c_uint,)
    counter.restype=ctypes.c_uint64
    def snapshot():return [counter(i) for i in range(4)]
    results=[]
    def copy(label,dst,src,eligible):
        before=snapshot()
        dst.copy_(src)
        after=snapshot()
        delta=[b-a for a,b in zip(before,after)]
        assert delta[:3] == (([1,0,1] if eligible else [0,1,0]) if cfg.switch=="1" else [0,0,0]),(label,delta)
        results.append(dict(label=label,eligible=eligible,producer_delta=delta[:3]))
    base,side=torch.npu.current_stream(),torch.npu.Stream()
    bits=((torch.arange(4800,dtype=torch.int64)*37+13)%65536-32768).short()
    bits[::111]=0x7fc1
    bits[1::111]=-32768
    bits[2::111]=0x7f80
    for stream in (base,side):
        with torch.npu.stream(stream):
            for offset in (0,64,128,256):
                s=bits.view(torch.bfloat16).npu()
                d=torch.zeros(4608,dtype=torch.bfloat16,device="npu")
                sv=s[offset:offset+4096]
                dv=d[64:4160]
                copy("owner-stream-"+str(stream.stream_id)+"-offset-"+str(offset),dv,sv,True)
                # Change producer view metadata and switch storage after the call.
                sv.resize_(1)
                sv.set_(torch.empty((16,),dtype=torch.bfloat16,device="npu"))
                del s,sv
                churn=[torch.zeros(4800,dtype=torch.bfloat16,device="npu") for _ in range(4)]
                want=torch.zeros(4608,dtype=torch.int16)
                want[64:4160]=bits[offset:offset+4096]
                assert torch.equal(d.cpu().view(torch.int16),want)
            src=bits.view(torch.bfloat16).npu()
            dst=torch.zeros_like(src)
            copy("short-fallback",dst[:4095],src[:4095],False)
            copy("strided-fallback",dst[:4096:2],src[:4096:2],False)
            assert torch.equal(dst[:4095].cpu().view(torch.int16),bits[:4095])
    s=bits.view(torch.bfloat16).npu()
    d=torch.empty_like(s)
    with torch.npu.stream(side):
        side.wait_stream(base)
        copy("cross-owner-fallback",d[:4096],s[64:4160],False)
        s.record_stream(side)
        d.record_stream(side)
    base.wait_stream(side)
    assert torch.equal(d[:4096].cpu().view(torch.int16),bits[64:4160])
    same=torch.zeros(8500,device="npu",dtype=torch.bfloat16)
    copy("alias-disjoint",same[4096:8192],same[:4096],False)
    before=snapshot()
    try:same[64:4160].copy_(same[:4096])
    except RuntimeError:assert snapshot()==before
    else:raise AssertionError("partial overlap accepted")
    torch.npu.synchronize()
    counts=snapshot()
    assert counts[0]==counts[2]==counts[3],counts
    cfg.output.write_text(json.dumps(dict(switch=cfg.switch,checks=results,counts=counts,
        overlap_error_preserved=True,library=library),indent=2)+"\n")


if __name__=="__main__":
    main()
