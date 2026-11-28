"""Run the staged Python and native range-copy paths and compare all observed state."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from stage import stage


def words(tensor):
    return tensor.detach().view(-1).to("cpu").view(torch.int16).tolist()


def trajectory(module,mode,device):
    os.environ["NPU_IPG_HOST_VIEWS"]=mode
    state,bucket,_=module.fixture(device,[6,5,4])
    z=state.z
    params=z.bit16_groups[0]
    dims=[(2,1,3),(5,),(1,4,1)]
    hooks=[]
    for i,(p,shape) in enumerate(zip(params,dims)):
        p.grad=None
        p.data=(torch.arange(p.numel(),device=device,dtype=torch.bfloat16).view(shape)+i)/7
        p.register_hook(lambda grad,i=i: hooks.append(("grad",i,words(grad))) or grad)
        p.register_post_accumulate_grad_hook(lambda param,i=i: hooks.append(("post",i,words(param.grad))))
    output=[]
    for step in range(5):
        for p in params:
            p.grad=None
        loss=sum(((p*(j+1)+step/8).float().square().sum()) for j,p in enumerate(params))
        loss.backward()
        if step==2:
            params[1].grad.view(-1)[-1]=float("inf")
        grads=[p.grad for p in params]
        buf=torch.cat([g.flatten() for g in grads])
        offset=0
        for g in grads:
            g.data=buf.narrow(0,offset,g.numel()).view_as(g).data
            offset+=g.numel()
        aliases=[g.detach() for g in grads]
        old_versions=[g._version for g in grads]
        bucket.buffer=[buf]
        z.grads_in_partition=None
        z.grads_in_partition_offset=0
        z.params_already_reduced=[False]*len(params)
        if device=="npu":
            z.reduction_stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(z.reduction_stream):
                assert state.try_bucket(bucket)
            torch.npu.current_stream().wait_stream(z.reduction_stream)
        else:
            assert state.try_bucket(bucket)
        dst=z.grads_in_partition
        assert all(p.grad is g and module.storage_id(g)==module.storage_id(dst) and g._version==v
                   for p,g,v in zip(params,grads,old_versions))
        assert all(module.storage_id(a)==module.storage_id(buf) for a in aliases)
        assert dst._version==len(grads) and all(z.params_already_reduced) and z.grads_in_partition_offset==dst.numel()
        finite=torch.isfinite(dst).all().item()
        assert bool(finite)==(step!=2)
        if finite:
            with torch.no_grad():
                for p in params: p.add_(p.grad,alpha=-.02)
        output.append(dict(loss=loss.item(), overflow=not finite, flat=words(dst),weights=[words(p) for p in params],
                           offsets=[g.storage_offset() for g in grads],strides=[g.stride() for g in grads],
                           grad_versions=[g._version for g in grads],dst_version=dst._version))
    return output,hooks,state.host_views


class Observe(TorchDispatchMode):
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        return func(*args,**(kwargs or {}))


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("build",type=Path)
    cli.add_argument("out",type=Path)
    cli.add_argument("--device",choices=["cpu","npu"],default="cpu")
    args=cli.parse_args()
    if args.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    torch.set_num_threads(1)
    os.environ["NPU_IPG_HOST_VIEWS_DIR"]=str(args.build.resolve())
    original=Path(__file__).resolve().parent.parent/"ipg_range_copy"
    with tempfile.TemporaryDirectory() as tmp:
        ws=Path(tmp)
        shutil.copyfile(original/"range_copy.py",ws/"range_copy.py")
        shutil.copyfile(original/"micro.py",ws/"range_micro.py")
        stage(ws)
        sys.path.insert(0,str(ws))
        import range_micro as m
        aa,ha,python=trajectory(m,"python",args.device)
        bb,hb,native=trajectory(m,"native",args.device)
        assert aa==bb and ha==hb and len(ha)==30
        assert python.published==0 and python.calls==native.calls==native.prepared==native.published==5
        positive,negative=m.small_checks(args.device)
        assert all(all(v[k] for k in ("all_words","source_values","all_views","offsets_equal","flags_equal"))
                   and v["dst_versions"]==[v["n"],v["n"]] for v in positive)
        assert all(v["rejected"] for v in negative.values())
        s,b,_=m.fixture("cpu",[4,4,4])
        with Observe():
            assert s.try_bucket(b)
        assert s.host_views.context_fallback==1 and s.host_views.published==0
        s,b,_=m.fixture("cpu",[4,4,4])
        flags=s.z.params_already_reduced
        class CustomList(list):
            pass
        s.z.params_already_reduced=CustomList(flags)
        assert s.try_bucket(b) and s.host_views.declined==1 and s.host_views.published==0
        s,b,_=m.fixture("cpu",[4,4,4])
        pre=[(m.storage_id(p.grad),p.grad._version) for p in s.z.bit16_groups[0]]
        _,(_,n,views)=s.prepare(b)
        dst=torch.empty(n-1,dtype=torch.bfloat16)
        assert s.host_views.module.prepare(dst,views,s.z.params_already_reduced) is None
        assert pre==[(m.storage_id(p.grad),p.grad._version) for p in s.z.bit16_groups[0]] and not any(s.z.params_already_reduced)
        result=dict(passed=True,device=args.device,torch=torch.__version__,library_sha256=native.sha,
            consecutive_steps=len(aa),skipped=[i for i,x in enumerate(aa) if x["overflow"]],
            equal_hook_observations=len(ha),loss=[x["loss"] for x in aa],
            native_published=native.published,semantic_checks=positive,range_fallback=negative,
            native_fallback=["TorchDispatchMode","custom flag list","invalid destination extent; no changes"])
        args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    main()
