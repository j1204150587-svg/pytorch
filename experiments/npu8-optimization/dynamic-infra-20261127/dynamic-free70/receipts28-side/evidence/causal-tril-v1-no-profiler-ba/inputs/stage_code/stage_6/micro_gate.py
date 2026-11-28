"""Compare the exact BF16 causal zeroing in the HSTU forward and backward."""
import argparse
import json
from pathlib import Path
import time

import torch


def original(scores):
    n=scores.shape[-1]
    mask=torch.tril(torch.ones(n,n,dtype=torch.bool,device=scores.device))
    return scores.masked_fill(~mask,0.) / float(n)


def tril(scores):
    return torch.tril(scores) / float(scores.shape[-1])


def execute(fn,x,dy):
    y=fn(x)
    dx,=torch.autograd.grad((y.float()*dy).sum(),x)
    return y.detach(),dx


def compare(x,y):
    a,b=x.cpu().contiguous(),y.cpu().contiguous()
    return int((a.view(torch.int16)!=b.view(torch.int16)).sum())


def run(device,repeats):
    sync=torch.npu.synchronize if device=="npu" else lambda:None
    params=[]
    tests=[]
    for b,l in ((1,19),(2,14),(2,22),(4,31),(4,32),(1,12)):
        rng=torch.Generator().manual_seed(1234+b+l)
        for kind in ("finite","nonfinite_signed_zero"):
            cpu=torch.randn((b,4,l,l),generator=rng,dtype=torch.bfloat16)
            gy=torch.randn((b,4,l,l),generator=rng)
            if kind=="nonfinite_signed_zero":
                # Include the diagonal, both halves, infinities, signed zero and noncanonical NaNs.
                vals=torch.tensor([0,-32768,32640,-128,32705,-47],dtype=torch.int16).view(torch.bfloat16)
                for i,v in enumerate(vals):
                    cpu[...,i,0]=v
                    cpu[...,0,i]=v
                    cpu[...,i,i]=v
                    # Test nonfinite upstream independently from the forward nonfinite.
                    gy[...,i,i-1]=v.float()
            x=cpu.to(device).requires_grad_(True)
            up=gy.to(device)
            a=execute(original,x,up)
            c=execute(tril,x,up)
            tests.append(dict(shape=[b,4,l,l],case=kind,output_bit_differences=compare(a[0],c[0]),
                              grad_bit_differences=compare(a[1],c[1])))
            if kind=="finite":params.append((x,up))
    passing=all(not (r["output_bit_differences"] or r["grad_bit_differences"]) for r in tests)
    pairs=[]
    if passing and repeats:
        for x,dy in params:
            execute(original,x,dy);execute(tril,x,dy)
        for order in ("AB","BA","AB","BA"):
            values={}
            for label in order:
                fn=original if label=="A" else tril
                sync();start=time.perf_counter_ns()
                for i in range(repeats):execute(fn,*params[i%len(params)])
                sync();values[label]=(time.perf_counter_ns()-start)/repeats/1000
            pairs.append(dict(order=order,calls=repeats,stock_us=values["A"],tril_us=values["B"],
                              pct=(1-values["B"]/values["A"])*100))
    return dict(torch=str(torch.__version__),device=device,cases=tests,passed=passing,timing=pairs)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cpu","npu"),required=True)
    parser.add_argument("--repeats",type=int,default=96)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    else:torch.set_num_threads(2)
    result=run(args.device,args.repeats)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"passed":result["passed"],"timing":result["timing"]}))
