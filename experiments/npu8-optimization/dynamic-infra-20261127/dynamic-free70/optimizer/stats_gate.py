"""Check flat stats against the corresponding 01/07 reductions on each device."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from flat_stats import nonfinite,norm_plan,row_norm_square


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cpu","npu"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    dev=args.device
    shape=[(45,),(2,8)]+[(64,64)]*69+[(2,3),(2,3),(1,)]
    gen=torch.Generator(device="cpu").manual_seed(1234)
    host=[torch.randn(s,generator=gen).to(torch.bfloat16) for s in shape]
    flat=torch.cat([x.reshape(-1) for x in host]).to(dev)
    shards=[x.view(s) for x,s in zip(flat.split([x.numel() for x in host]),shape,strict=True)]
    stage=SimpleNamespace(PIPE_REPLICATED="ds_pipe_replicated",is_model_parallel_parameter=lambda p:False)
    zero=SimpleNamespace(model_parallel_rank=0)
    plan=norm_plan(shards,stage,zero,256*1024,64)
    original=[]
    begin=0
    for offset,shape,count,width in plan:
        subset=shards[begin:begin+count]
        if count==1:
            norm=torch.linalg.vector_norm(subset[0].data.double().detach(),ord=2).reshape(1)
        else:
            packed=torch.stack(subset).view(count,-1)
            norm=torch.linalg.vector_norm(packed.double().detach(),ord=2,dim=1)
        original.append(norm)
        begin+=count
    expected_norm=torch.cat(original).square().sum().float()
    actual_norm=row_norm_square(flat,plan)
    if not torch.equal(expected_norm,actual_norm):
        raise ValueError(f"norm mismatch: {expected_norm},{actual_norm}")
    checks=[]
    def check(label,value,expected):
        result=nonfinite(value)
        ref=value.float()
        ref=(ref.isnan()|ref.isinf()).any().reshape(1)
        if result.dtype!=torch.bool or result.shape!=(1,) or bool(result)!=expected or not torch.equal(ref,result):
            raise ValueError(label)
        checks.append(dict(case=label,invalid=expected))
    check("finite",flat,False)
    check("max_values_with_overflowing_sum",torch.full((1000,),torch.finfo(torch.bfloat16).max,device=dev,dtype=torch.bfloat16),False)
    for label,bad in (("nan",float("nan")),("posinf",float("inf")),("neginf",-float("inf"))):
        for i in (0,flat.numel()//2,-1):
            x=flat.clone()
            x[i]=bad
            check(label+str(i),x,True)
    args.output.write_text(json.dumps(dict(device=dev,passed=True,group_plan=plan,
        squared_norm=float(actual_norm),checked_nonfinite=checks),indent=2))


if __name__=="__main__":
    main()
