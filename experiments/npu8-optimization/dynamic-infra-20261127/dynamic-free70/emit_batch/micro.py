"""Test actual eager CANN commands for 1,024 independent tables; also time the compact ATen alternative."""
import argparse
import ctypes as C
import json
from pathlib import Path
from time import perf_counter
import torch
import torch_npu


def library(path):
    dll = C.CDLL(str(path.resolve()))
    ptr = C.c_void_p
    for name, args, ret in (("new", [C.c_int64]*3, ptr),
                            ("plan", [ptr]*4+[C.POINTER(C.c_uint64)], C.c_int),
                            ("submit", [ptr]*6, C.c_int), ("close",[ptr], C.c_int)):
        fn = getattr(dll, "packet_"+name)
        fn.argtypes, fn.restype = args, ret
    return dll


def check(status):
    if status:
        raise RuntimeError(f"ACLNN packet returned {status}")


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    torch.npu.set_device(0)
    lib=library(args.library)
    op=torch.ops.aten.embedding_dense_backward.default
    rows=[]
    for b,l in ((1,19),(2,14),(2,22)):
        torch.manual_seed(1234)
        ids=torch.randint(64,(b,l,1024,4)).npu()
        grad=torch.randn(b,l,4,64).bfloat16().npu()
        outputs=torch.empty((1024,64,64),dtype=torch.bfloat16,device="npu")
        expected=torch.stack([op(grad,ids[:,:,t,:],64,-1,False) for t in range(1024)])
        # Independent-table ATen oracle and separate compact implementation.
        idx=(ids.permute(2,0,1,3)+(torch.arange(1024,device="npu")*64).view(-1,1,1,1)).contiguous()
        dense=grad.unsqueeze(0).expand(1024,-1,-1,-1,-1).contiguous()
        stream=torch.npu.current_stream()
        stream.synchronize()
        handle=lib.packet_new(b,l,1024)
        amount=C.c_uint64()
        start=perf_counter()
        check(lib.packet_plan(handle,grad.data_ptr(),ids.data_ptr(),outputs.data_ptr(),C.byref(amount)))
        plan_ms=(perf_counter()-start)*1000
        scratch=torch.empty(max(1,amount.value),dtype=torch.uint8,device="npu")
        ptr=stream.npu_stream
        args2=(handle,grad.data_ptr(),ids.data_ptr(),outputs.data_ptr(),scratch.data_ptr(),ptr)
        check(lib.packet_submit(*args2))
        stream.synchronize()
        valid=torch.equal(expected.cpu(),outputs.cpu())
        if not valid:
            raise RuntimeError("gradient bit mismatch")
        measures=[]
        for order in ("ABCCBA","CBAABC"):
            for variant in order:
                stream.synchronize()
                start=perf_counter()
                if variant=="A":
                    vals=[op(grad,ids[:,:,t,:],64,-1,False) for t in range(1024)]
                elif variant=="B":
                    check(lib.packet_submit(*args2))
                else:
                    vals=op(dense,idx,1024*64,-1,False)
                host=(perf_counter()-start)*1000
                stream.synchronize()
                measures.append(dict(variant=variant,host_ms=host,total_ms=(perf_counter()-start)*1000))
        count=lib.packet_close(handle)
        rows.append(dict(shape=[b,l],workspace=amount.value,builds=16,prepare_ms=plan_ms,
                         actual_table_requests=count,bitwise_equal=valid,measurements=measures))
        args.output.write_text(json.dumps(dict(variants={"A":"stock table loop","B":"CANN repeatable packet",
            "C":"compact global embedding, prepared data"},rows=rows),indent=2))


if __name__=="__main__":
    main()
