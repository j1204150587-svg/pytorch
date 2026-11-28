"""Check protection bytes, owner rebinding, two streams and fallback for 128-byte residuals."""

import argparse
import ctypes
import json
import os
from pathlib import Path


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("output",type=Path)
    cli.add_argument("--switch",choices=("0","1"),required=True)
    args=cli.parse_args()
    os.environ.update(NPU_INPLACE_COPY_REBASE=args.switch,NPU_DYNAMIC_REBASE_COUNT="1")
    import torch
    import torch_npu
    torch.npu.set_device(0)
    libs={q.split()[-1] for q in Path("/proc/self/maps").read_text().splitlines()
          if q.split()[-1].endswith("/libtorch_npu.so")}
    lib,=libs
    counter=ctypes.CDLL(lib).npu_inplace_copy_rebase_counter
    counter.argtypes=(ctypes.c_uint,)
    counter.restype=ctypes.c_uint64
    def counts():return [counter(i) for i in range(4)]
    results=[]
    def copy(label,dst,src,eligible):
        before=counts()
        dst.copy_(src)
        delta=[b-a for a,b in zip(before,counts())]
        if args.switch=="1":
            assert delta[:3]==([1,0,2] if eligible else [0,1,0]),(label,delta)
        else:
            assert delta==[0,0,0,0],(label,delta)
        results.append(dict(label=label,eligible=eligible,delta=delta))
    base,other=torch.npu.current_stream(),torch.npu.Stream()
    def bits(n,seed):
        # Include signed zero, infinity and NaN encodings around interior/sentinel boundaries.
        flat=((torch.arange(n,dtype=torch.int64)*343+seed*91)%65536-32768).to(torch.int16)
        flat[::127]=0x7f80
        flat[1::127]=0x7fc1
        flat[2::127]=-32768
        return flat
    for repeat in range(4):
        for offset_src in (0,64,256,320):
            for slot in (0,1):
                offset_dst=64+slot*512+repeat*256
                length=4096
                sb=bits(length+offset_src+(0 if slot else 128),repeat+slot+3)
                db=bits(offset_dst+length+(128 if slot else 0),repeat+slot+31)
                s=sb.view(torch.bfloat16).npu()
                d=db.view(torch.bfloat16).npu()
                expected=db.clone()
                expected[offset_dst:offset_dst+length]=sb[offset_src:offset_src+length]
                sv=s.narrow(0,offset_src,length)
                dv=d.narrow(0,offset_dst,length)
                if slot:
                    with torch.npu.stream(other):
                        other.wait_stream(base)
                        copy(f"{repeat}/{offset_src}/{slot}",dv,sv,True)
                        s.record_stream(other)
                        d.record_stream(other)
                    base.wait_stream(other)
                else:
                    copy(f"{repeat}/{offset_src}/{slot}",dv,sv,True)
                assert torch.equal(d.cpu().view(torch.int16),expected)
                assert torch.equal(s.cpu().view(torch.int16),sb)
    for name,off_src,off_dst,n,dtype in (("source-residual64",32,64,4096,torch.bfloat16),
            ("dest-residual0",0,0,4096,torch.bfloat16),("length",0,64,4095,torch.bfloat16),
            ("dtype",0,64,4096,torch.float32)):
        h=torch.arange(4400,dtype=torch.int32).remainder(239).to(dtype)
        z=torch.zeros_like(h)
        s,d=h.npu(),z.npu()
        z[off_dst:off_dst+n]=h[off_src:off_src+n]
        copy(name,d[off_dst:off_dst+n],s[off_src:off_src+n],False)
        assert torch.equal(d.cpu(),z)
    h=torch.arange(9000,dtype=torch.int32).remainder(199).bfloat16()
    d=h.npu()
    h[4160:8256]=h[:4096]
    copy("shared-storage-disjoint",d[4160:8256],d[:4096],False)
    assert torch.equal(d.cpu(),h)
    before=counts()
    try:d[64:4160].copy_(d[:4096])
    except RuntimeError:assert counts()==before and torch.equal(d.cpu(),h)
    else:raise AssertionError("partial overlap unexpectedly accepted")
    args.output.write_text(json.dumps(dict(switch=args.switch,checks=results,overlap_error_preserved=True,
                                            bitwise=True,library=lib,counters_final=counts()),indent=2)+"\n")


if __name__=="__main__":
    main()
