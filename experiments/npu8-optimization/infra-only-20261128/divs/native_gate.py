"""Compare functional native Div against the original out= path and verify native dispatch counters."""
import argparse
import ctypes
import json
import os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule",type=Path,required=True)
    ap.add_argument("--output","--out",type=Path,required=True)
    args = ap.parse_args()
    os.environ["NPU_HSTU_COMPACT_DIV"] = "1"
    import torch
    import torch_npu
    from probe import source, word_diff, save
    torch.npu.set_device(0)
    schedule = json.loads(args.schedule.read_text())
    gen = torch.Generator().manual_seed(83125)
    count = ctypes.CDLL(str(Path(torch_npu.__file__).parent/"lib/libtorch_npu.so")).npu_hstu_compact_div_counter
    count.argtypes = [ctypes.c_uint]
    count.restype = ctypes.c_uint64
    def counters():
        return [count(i) for i in range(4)]
    def delta(a,b):
        return [y-x for x,y in zip(a,b)]
    def ref(x,scalar):
        out = torch.empty(x.size(),dtype=x.dtype,device=x.device)
        return torch.ops.aten.div.out(x,torch.tensor(scalar),out=out)
    def compare(x,scalar,op,eligible):
        expected = ref(x,scalar)
        original = x.cpu().clone()
        start = counters()
        y = op(x,scalar)
        change = delta(start,counters())
        y2 = op(x,scalar)
        dif = word_diff(y,expected)
        layout = y.size()==expected.size() and y.stride()==expected.stride() and y.dtype==expected.dtype
        distinct = len({x.untyped_storage().data_ptr(), y.untyped_storage().data_ptr(), y2.untyped_storage().data_ptr()})==3
        y.zero_()
        alias = word_diff(x,original)["different"] + word_diff(expected,y2)["different"]
        return dict(shape=list(x.size()), offset=x.storage_offset(), scalar=scalar, eligible=eligible,
                    differences=dif, counters=change, layout=layout, distinct=distinct, alias_change=alias,
                    passed=sum(change[:2])==int(eligible) and dif["different"]==0 and layout and distinct and alias==0)
    paths = [("operator",lambda x,s:x/s), ("scalar",torch.ops.aten.div.Scalar),
             ("cpu_scalar_tensor",lambda x,s:torch.ops.aten.div.Tensor(x,torch.tensor(s)))]
    report = dict(schedule=schedule,checks=[],grads=[])
    streams = [torch.npu.current_stream(),torch.npu.Stream()]
    for j,stream in enumerate(streams):
        with torch.npu.stream(stream):
            for i in (range(len(schedule)) if j==0 else (0,4,5,len(schedule)-1)):
                b,l=schedule[i]
                x=source(b,l,0 if j==0 else 7,gen,i in (0,4,5,len(schedule)-1))
                for label,op in paths:
                    report["checks"].append(dict(compare(x,4,op,True),op=label,stream=j))
            torch.npu.synchronize()
    # Ineligible division modes, layout and feature width share the original path.
    x=source(2,14,0,gen)
    for x2,s in ((x,3), (x.contiguous(),4), (x[:,:,:3],4), (x[:,:,:,:32],4), (source(1,8,0,gen),4)):
        report["checks"].append(dict(compare(x2,s,paths[0][1],False),op="fallback",stream=0))
    for i in (0,4,5,len(schedule)-1):
        b,l=schedule[i]
        x=source(b,l,0,gen)[:,:,:1].contiguous()
        u=x.clone().detach().requires_grad_()
        v=x.clone().detach().requires_grad_()
        upstream=torch.randn(b,l,4,64,generator=gen).bfloat16().npu()
        start=counters()
        out=u.expand(b,l,4,64)/4
        change=delta(start,counters())
        stock=v.expand(b,l,4,64).contiguous()/4
        out.backward(upstream)
        stock.backward(upstream)
        grad=word_diff(u.grad,v.grad)
        report["grads"].append(dict(shape=[b,l],counters=change,gradient=grad,
                                    passed=grad["different"]==0 and sum(change[:2])==1))
    report["total_counters"]=counters()
    report["passed"]=all(row["passed"] for row in report["checks"]+report["grads"])
    save(args.output,report)
    if not report["passed"]:
        raise SystemExit(2)


if __name__=="__main__":
    main()
