"""Localize the first full-candidate finite BF16 differences on CPU."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from attention_case import cpu_full_forward
from micro import bits, build


def diagnose(case):
    b,l = case["shape"]
    q,k,v,dy = build(b,l,"cpu","finite",case["seed"])
    raw = q @ k.transpose(-1,-2)
    x = raw*.25
    s = F.silu(x).masked_fill(~torch.ones(l,l,dtype=torch.bool).tril(),0.) / float(l)
    y = s@v
    dx0,dr0,dq0,ds0 = torch.autograd.grad(y,(x,raw,q,s),dy)
    xx,ss,yy = cpu_full_forward(q,k,v)
    sig = xx.float().sigmoid()
    ds = (dy.float() @ v.float().transpose(-2,-1)).bfloat16()
    g = torch.tril((ds.float()/float(l)).bfloat16()).float()
    dx = (g*(sig*(1.+xx.float()*(1.-sig)))).bfloat16()
    dr = (dx.float()*.25).bfloat16()
    probes = [
        ("QKt",raw,(q.float()@k.float().transpose(-1,-2)).bfloat16()),
        ("scaled_QKt",x,xx),("normalized_score",s,ss),("Y",y,yy),
        ("d_score",ds0,ds),("d_silu_input",dx0,dx),("d_raw",dr0,dr),
        ("dQ_float_matmul_then_cast",dq0,(dr.float()@k.float()).bfloat16()),
        ("dQ_same_BF16_ATen_matmul",dq0,dr@k),
    ]
    return {"shape":[b,l],"seed":case["seed"],"words":{n:bits(a,c) for n,a,c in probes}}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("input",type=Path)
    cli.add_argument("output",type=Path)
    args = cli.parse_args()
    torch.set_num_threads(1)
    cases = json.loads(args.input.read_text(encoding="utf-8"))["cases"]
    probes = [diagnose(c) for c in cases if c["case"] == "finite" and not c["passed"]]
    args.output.write_text(json.dumps(probes,indent=2)+"\n",encoding="utf-8")
