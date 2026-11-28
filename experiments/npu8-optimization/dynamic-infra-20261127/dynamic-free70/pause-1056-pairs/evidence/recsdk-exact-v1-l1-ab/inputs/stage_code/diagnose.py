"""Localize a native finite-case discrepancy by projecting attention weights onto basis V columns."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"causal_silu"))
from micro import bits,build,save


def run(q,k,v):
    l=q.size(2)
    args=[x.transpose(1,2) for x in (q,k,v)]
    return torch.ops.mxrec.hstu_dense(*args,None,None,0,l,0.).transpose(1,2)


def case(b,l,sample):
    q,k,v,dy=build(b,l,"npu","finite",250+sample)
    shift=(q@k.transpose(-1,-2))*.25
    valid=torch.ones(l,l,dtype=torch.bool,device=q.device).tril()
    ref_score=F.silu(shift).masked_fill(~valid,0.)/float(l)
    with torch.no_grad():
        chunks=[]
        for start in range(0,l,16):
            width=min(l-start,16)
            select=torch.zeros_like(v)
            select[:,:,start:start+width,:width]=torch.eye(width,device=q.device,dtype=q.dtype)
            chunks.append(run(q,k,select)[...,:width])
        observed=torch.cat(chunks,dim=-1)
        out=run(q,k,v)
        return dict(shape=[b,l], finite=True,
                    recovered_score_vs_eager=bits(ref_score,observed),
                    native_y_vs_eager_score_matmul=bits(out,ref_score@v),
                    native_y_vs_recovered_score_matmul=bits(out,observed@v))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--schedule",type=Path,required=True)
    p.add_argument("--gate",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    a=p.parse_args()
    torch.npu.set_device(0)
    torch.ops.load_library(str(a.library.resolve(strict=True)))
    schedule=json.loads(a.schedule.read_text())
    original=json.loads(a.gate.read_text())
    bad={tuple(c["shape"]) for c in original["cases"] if c["case"]=="finite" and any(v["differing_bf16_words"] for v in c["differences"].values())}
    report=dict(method="Recover finite W by Y_native(q,k,V_basis), compare W, W@V, recovered_W@V",cases=[])
    for i,(b,l) in enumerate(schedule):
        if (b,l) in bad:
            report["cases"].append(case(b,l,i))
            save(report,a.out)
    report["complete"]=True
    save(report,a.out)
