"""Isolated NPU comparison against the original per-table embedding/mean gradient."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu

from dense_grad import grouped_dense_backward


def case(batch,length,tables,kind,generator):
    ids = torch.randint(64,(batch,length,tables,4),generator=generator)
    ids[...,1] = ids[...,0]
    ids[...,2] = 0
    ids[...,3] = 63
    grad = torch.randn((batch,length,64),generator=generator).to(torch.bfloat16)
    if kind == "ill_conditioned":
        # Deliberate cancellation and a value at a BF16 tie.
        grad.reshape(-1,64)[:3,0] = torch.tensor([2**24,1,-2**24],dtype=grad.dtype)
        grad.reshape(-1,64)[:3,1] = torch.tensor([1,2**-8,2**-23],dtype=grad.dtype)
        ids.reshape(-1,tables,4)[:3,:,0] = 0
    if kind == "nonfinite":
        grad.reshape(-1,64)[:3,0] = torch.tensor([float("inf"),-float("inf"),float("nan")],dtype=grad.dtype)
        grad.reshape(-1,64)[0,1] = float("inf")
    ids = ids.npu()
    grad = grad.npu()
    target = torch.empty((64,64),device=grad.device,dtype=grad.dtype,requires_grad=True)
    expected = []
    for t in range(tables):
        out = F.embedding(ids[:,:,t,:],target).mean(dim=-2)
        expected.append(torch.autograd.grad(out,target,grad)[0])
    want = torch.stack(expected).cpu()
    got = grouped_dense_backward(ids,grad).cpu()
    same = (want.view(torch.int16)==got.view(torch.int16)) | (want.isnan() & got.isnan())
    finite = want.isfinite() & got.isfinite()
    max_diff = (want[finite].float()-got[finite].float()).abs().max().item() if finite.any() else 0
    return dict(shape=[batch,length],tables=tables,kind=kind,
                elements=same.numel(),different=int((~same).sum()),
                first_differences=(~same).nonzero()[:5].tolist(),max_finite_absolute_difference=max_diff)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--tables",type=int,default=1024)
    args = parser.parse_args()
    torch.npu.set_device(0)
    gen = torch.Generator(device="cpu").manual_seed(1234)
    rows = [case(b,l,args.tables,"finite",gen) for b,l in ((1,19),(2,14),(2,22),(4,31))]
    rows.extend(case(2,14,16,k,gen) for k in ("ill_conditioned","nonfinite"))
    passed = all(x["different"]==0 for x in rows)
    args.output.write_text(json.dumps(dict(torch=torch.__version__,cases=rows,passed=passed),indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
