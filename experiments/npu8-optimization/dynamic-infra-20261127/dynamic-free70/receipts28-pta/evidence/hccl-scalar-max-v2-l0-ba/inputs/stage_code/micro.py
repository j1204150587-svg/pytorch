"""Exercise real rank-one ProcessGroupHCCL across streams, Work/Future and exceptional FP32 bits."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import tempfile
import torch
import torch.distributed as dist


def words(t):
    return t.detach().cpu().contiguous().view(torch.int32).tolist() if t.dtype==torch.float32 else t.detach().cpu().tolist()


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    import torch_npu
    torch.npu.set_device(0)
    records=[]
    with tempfile.TemporaryDirectory() as tmp:
        store=Path(tmp)/"group"
        dist.init_process_group("hccl",init_method=store.as_uri(),world_size=1,rank=0,timeout=timedelta(seconds=45))
        inputs=[
            ("imax",torch.tensor([-2147483648],dtype=torch.int32),dist.ReduceOp.MAX),
            ("byte",torch.tensor([255],dtype=torch.uint8),dist.ReduceOp.MAX),
            ("bool",torch.tensor([True]),dist.ReduceOp.MAX),
            ("vector",torch.tensor([-1,2],dtype=torch.int32),dist.ReduceOp.MAX),
            ("intsum",torch.tensor([213328925],dtype=torch.int32),dist.ReduceOp.SUM)]
        for bits in (0x80000000,0x00000000,0x00000001,0x80000001,0x7f800000,0xff800000,0x7fc12001,0x7f812345,0x3f987654):
            u=bits if bits<2**31 else bits-2**32
            inputs.append((f"fp32_{bits:08x}",torch.tensor([u],dtype=torch.int32).view(torch.float32),dist.ReduceOp.SUM))
        device=torch.device("npu",0)
        producer=torch.npu.Stream(device=device)
        consumer=torch.npu.Stream(device=device)
        for name,host,op in inputs:
            value=torch.empty_like(host,device=device)
            producer.wait_stream(torch.npu.current_stream(device))
            with torch.npu.stream(producer):
                value.copy_(host.to(device))
                w=dist.all_reduce(value,op=op,async_op=True)
            assert w is not None
            f=w.get_future()
            # wait() supplies a dependency for the current consumer stream.
            with torch.npu.stream(consumer):
                wait_ok=w.wait()
                snapshot=value.clone()
            torch.npu.current_stream(device).wait_stream(consumer)
            out=f.wait()
            if not isinstance(out,list) or len(out)!=1:
                raise RuntimeError("unexpected Work Future payload")
            actual=words(snapshot)
            observed=words(out[0])
            assert wait_ok and w.is_completed() and actual==observed
            expected=words(host)
            records.append(dict(name=name,source=expected,actual=actual,identity=expected==actual,
                                future_dtype=str(out[0].dtype),aliases_future=out[0].data_ptr()==value.data_ptr(),
                                work_done=w.is_completed(),wait_ok=wait_ok))
        dist.destroy_process_group()
    native_counts=None
    if "NPU_HCCL_RANK1_SCALAR" in os.environ:
        from counts import load
        path,fn=load()
        native_counts=dict(library=path,hits=fn(0),health_fallback=fn(1))
        if native_counts["hits"]!=(3 if os.environ["NPU_HCCL_RANK1_SCALAR"]=="intmax" else 0) or fn(1)!=0:
            raise RuntimeError(f"unexpected real native activation {native_counts}")
    args.output.write_text(json.dumps(dict(torch=torch.__version__,torch_npu=torch_npu.__version__,
        option=os.getenv("NPU_HCCL_RANK1_SCALAR","0"),native_counts=native_counts,records=records),indent=2)+"\n")
