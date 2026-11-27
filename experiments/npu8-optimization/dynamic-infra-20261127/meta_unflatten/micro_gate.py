"""Compare the eligible mixed-device helper and stock fallback on the measured NPU wheel."""
import argparse
import json
import os
from pathlib import Path
import torch
import torch_npu
from torch.utils._python_dispatch import TorchDispatchMode
from torch_npu._dynamic_meta_unflatten import counters


def signature(t):
    return (tuple(t.shape),tuple(t.stride()),t.storage_offset(),t.dtype,t.device.type)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    torch.npu.set_device(0)
    helper=torch._utils._unflatten_dense_tensors
    original=torch._C._nn.unflatten_dense_tensors
    mode=os.getenv("NPU_DYNAMIC_META_UNFLATTEN","0")
    shapes=[(),(1,3),(2,2),(2,1,2)]*9
    refs=[torch.empty(s,device="meta",dtype=torch.bfloat16) for s in shapes]
    n=sum(t.numel() for t in refs)
    seen=[]

    def make(dtype=torch.bfloat16):
        return torch.arange(n+64,device="npu:0").to(dtype).narrow(0,9,n+7)

    def compared(label,flat,templates,eligible=False):
        start=counters()
        expected=original(flat,templates)
        actual=helper(flat,templates)
        end=counters()
        if type(actual) is not type(expected) or len(actual)!=len(expected):
            raise ValueError(f"{label}: heterogeneous return type or count")
        for x,y in zip(actual,expected,strict=True):
            if signature(x)!=signature(y) or not torch.equal(x,y):
                raise ValueError(f"{label}: different view/value")
            if x.numel():
                storage=flat.untyped_storage()._cdata
                if x.untyped_storage()._cdata!=storage or y.untyped_storage()._cdata!=storage:
                    raise ValueError(f"{label}: missing output storage alias")
                if (x._base is None)!=(y._base is None):
                    raise ValueError(f"{label}: view tracking differs from stock")
                if x._base is not None and x._base.untyped_storage()._cdata!=y._base.untyped_storage()._cdata:
                    raise ValueError(f"{label}: view bases differ")
        wanted=int(mode=="1" and eligible)
        if end["eligible"]-start["eligible"]!=wanted:
            raise ValueError(f"{label}: unexpected activation {start} -> {end}")
        seen.append(dict(test=label,eligible=wanted))
        return actual

    for shift in (0,3,7):
        flat=make()[shift:]
        outputs=compared(f"offset{shift}",flat,refs,True)
        again=compared(f"fresh_view{shift}",flat,refs,True)
        if any(x._cdata==y._cdata for x,y in zip(outputs,again)):
            raise ValueError("output TensorImpl unexpectedly reused")
        version=flat._version
        old=outputs[0].clone()
        outputs[0].add_(1)
        if flat._version!=version+1 or not torch.equal(again[0],old+1):
            raise ValueError("version/alias propagation")
    # Empty templates have native empty-storage semantics, including the shape.
    compared("empty",make(),[torch.empty(0,device="meta",dtype=torch.bfloat16)]+refs)
    compared("mixed_device",make(),refs[:-1]+[torch.empty(shapes[-1],device="npu:0",dtype=torch.bfloat16)])
    compared("small_list",make(),refs[:8])
    compared("wrong_dtype",make(torch.float32),refs)
    compared("strided_flat",make().repeat_interleave(2)[::2],refs)
    sparse=torch.sparse_coo_tensor(size=(2,2),device="meta",dtype=torch.bfloat16)
    compared("sparse_template",make(),[sparse]+refs[1:])
    with torch.inference_mode():
        compared("inference",make().clone(),refs)

    grad=make().detach().requires_grad_(True)
    ref=grad.detach().clone().requires_grad_(True)
    outs=compared("reverse_ad",grad,refs)
    for x,items in ((grad,outs),(ref,original(ref,refs))):
        sum((j+1)*t.float().sum() for j,t in enumerate(items)).backward()
    if not torch.equal(grad.grad,ref.grad):
        raise ValueError("reverse gradients differ")
    with torch.autograd.forward_ad.dual_level():
        primal=make()
        tangent=torch.ones_like(primal)
        dual=torch.autograd.forward_ad.make_dual(primal,tangent)
        outs=compared("forward_ad",dual,refs)
        for x,y in zip(outs,original(tangent,refs),strict=True):
            if not torch.equal(torch.autograd.forward_ad.unpack_dual(x).tangent,y):
                raise ValueError("forward tangent differs")

    class Noop(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            return func(*args,**(kwargs or {}))

    with Noop():
        compared("dispatch_mode",make(),refs)
    bad=make()[:n-1]
    failures=[]
    for f in (original,helper):
        try:
            f(bad,refs)
        except RuntimeError as e:
            failures.append(str(e))
    if len(failures)!=2 or failures[0]!=failures[1]:
        raise ValueError(f"native short-buffer error changed: {failures}")
    args.output.write_text(json.dumps(dict(mode=mode,tests=seen,counters=counters(),
        aten_dispatch_table=torch._C._dispatch_dump_table("aten::unflatten_dense_tensors"),
        stock_short_buffer_error=failures[0]),indent=2))


if __name__=="__main__":
    main()
