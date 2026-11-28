"""Exercise full-range values, publication, mutation counts, and pre-submit fallbacks."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from range_copy import BucketCopy,storage_id


def fixture(device,sizes,offset=7):
    n=sum(sizes)
    raw=(torch.arange(n+offset+3)%211-100).bfloat16()
    raw[offset+1]=float("nan")
    raw[offset+2]=float("-inf")
    raw[offset+3]=-0.
    ipg=raw.to(device).narrow(0,offset,n)
    params=[]
    cursor=0
    for size in sizes:
        p=torch.nn.Parameter(torch.empty(size,device=device,dtype=torch.bfloat16))
        p.grad=ipg.narrow(0,cursor,size)
        params.append(p)
        cursor+=size
    ids={id(p):i for i,p in enumerate(params)}
    z=SimpleNamespace(partition_gradients=True,contiguous_gradients=True,cpu_offload=False,
            zenflow=False,use_grad_accum_attribute=False,params_in_partition=[params],bit16_groups=[params],
            dtype=torch.bfloat16,grads_in_partition=None,grads_in_partition_offset=0,round_robin_gradients=False,
            overlap_comm=True,reduction_stream=torch.npu.Stream() if device=="npu" else None,
            params_already_reduced=[False]*len(params),is_param_in_current_partition=[True]*len(params),
            get_param_id=lambda p:ids[id(p)],get_gradient_for_reduction=lambda p:p.grad)
    bucket=SimpleNamespace(buffer=[ipg],index=0,params=[(0,j,j) for j in range(len(params))],elements=n)
    state=BucketCopy(z,"range")
    state.group_id=id(params)
    state.group_elements=n
    return state,bucket,raw


def stock(z,b):
    dst=torch.empty(b.elements,device=b.buffer[0].device,dtype=z.dtype)
    cursor=0
    for g,i,pid in b.params:
        p=z.bit16_groups[g][i]
        grad=p.grad
        z.params_already_reduced[pid]=True
        slot=dst.view(-1).narrow(0,cursor,p.numel())
        slot.copy_(grad.view(-1))
        grad.data=slot.data.view_as(grad)
        cursor+=p.numel()
    z.grads_in_partition=dst
    z.grads_in_partition_offset=cursor


def small_checks(device):
    result=[]
    for sizes in ([3,1,9,4,17],[2+i%5 for i in range(1052)]):
        state, bucket, source=fixture(device,sizes)
        a, b, _=fixture(device,sizes)
        args=[(a,b,False),(state,bucket,True)]
        for owner,work,treat in args:
            cm=nullcontext()
            if device=="npu":
                owner.z.reduction_stream.wait_stream(torch.npu.current_stream())
                cm=torch.npu.stream(owner.z.reduction_stream)
            with cm:
                if treat:
                    assert owner.try_bucket(work)
                else:
                    stock(owner.z,work)
        if device=="npu": torch.npu.synchronize()
        ad=a.z.grads_in_partition
        bd=state.z.grads_in_partition
        same=torch.equal(ad.cpu().view(torch.int16),bd.cpu().view(torch.int16))
        same_source=torch.equal(bd.cpu().view(torch.int16),source[7:7+sum(sizes)].view(torch.int16))
        aparams=a.z.bit16_groups[0]
        bparams=state.z.bit16_groups[0]
        views=all((p.grad.shape,p.grad.stride(),p.grad.storage_offset(),p.grad._version)==
                  (q.grad.shape,q.grad.stride(),q.grad.storage_offset(),q.grad._version)
                  and storage_id(q.grad)==storage_id(bd) for p,q in zip(aparams,bparams))
        result.append(dict(n=len(sizes),all_words=same,source_values=same_source,all_views=views,
                           dst_versions=[ad._version,bd._version],
                           offsets_equal=a.z.grads_in_partition_offset==state.z.grads_in_partition_offset,
                           flags_equal=a.z.params_already_reduced==state.z.params_already_reduced))
    bad={}
    for kind in ("reverse","foreign","seen","short","existing","wrong_stream"):
        if kind=="wrong_stream" and device=="cpu": continue
        p,b,_=fixture(device,[4,4,4])
        if kind=="reverse": b.params.reverse()
        if kind=="foreign": p.z.bit16_groups[0][0].grad=p.z.bit16_groups[0][0].grad.clone()
        if kind=="seen": p.z.params_already_reduced[1]=True
        if kind=="short": b.params.pop()
        if kind=="existing": p.z.grads_in_partition=b.buffer[0]
        prior=p.z.grads_in_partition
        guards=[(storage_id(g.grad),g.grad.storage_offset(),g.grad._version) for g in p.z.bit16_groups[0]]
        cm=torch.npu.stream(p.z.reduction_stream) if device=="npu" and kind!="wrong_stream" else nullcontext()
        with cm: accepted=p.try_bucket(b)
        after=[(storage_id(g.grad),g.grad.storage_offset(),g.grad._version) for g in p.z.bit16_groups[0]]
        bad[kind]=dict(rejected=not accepted and p.z.grads_in_partition is prior and guards==after,
                       reason=p.attempts[-1]["reason"])
    return result,bad


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--device",choices=("cpu","npu"),required=True)
    cli.add_argument("--out",type=Path,required=True)
    a=cli.parse_args()
    if a.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    torch.set_num_threads(1)
    positive,negative=small_checks(a.device)
    ok=all(all(r[k] for k in ("all_words","source_values","all_views","offsets_equal","flags_equal"))
           and r["dst_versions"]==[r["n"],r["n"]] for r in positive) and all(r["rejected"] for r in negative.values())
    a.out.write_text(json.dumps(dict(device=a.device,torch=torch.__version__,passed=ok,positive=positive,
                                     negative=negative),indent=2)+"\n")
    raise SystemExit(0 if ok else 1)
