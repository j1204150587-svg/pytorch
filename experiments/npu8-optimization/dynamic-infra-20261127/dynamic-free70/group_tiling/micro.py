"""Measure equal-work table-group sizes for stock ACLNN dense embedding gradients on dynamic inputs."""
import argparse
import json
from pathlib import Path
from statistics import median
import time

import torch
import torch_npu


def layout(batch, length, width, tables, rows, device, seed):
    gen = torch.Generator().manual_seed(seed)
    ids = torch.randint(rows, (batch, length, tables, 4), generator=gen).to(device)
    diff = torch.randn(batch, length, width, generator=gen).bfloat16().to(device)
    return ids, diff


def inputs(source, upstream, group, rows):
    t = source.shape[2]
    b, l = source.shape[:2]
    ids = source.permute(2, 0, 1, 3).reshape(t // group, group, b, l, 4)
    offsets = (torch.arange(group, device=source.device) * rows).view(1, group, 1, 1, 1)
    ids = (ids + offsets).contiguous()
    grad = (upstream.unsqueeze(0).unsqueeze(-2).expand(group, b, l, 4, upstream.shape[-1]) / 4).contiguous()
    return ids, grad


def evaluate(source, upstream, group, rows):
    ids, grad = inputs(source, upstream, group, rows)
    op = torch.ops.aten.embedding_dense_backward.default
    block = [op(grad, x, group*rows, -1, False) for x in ids.unbind(0)]
    return block[0] if len(block)==1 else torch.cat(block, dim=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    torch.npu.set_device(0)
    rows=[]
    groups=(16, 64, 256, 1024)
    for b,l in ((1,19),(2,14),(2,22),(4,31)):
        x,g=layout(b,l,64,1024,64,"npu",1234)
        ref=evaluate(x,g,1024,64).cpu()
        for group in groups:
            out=evaluate(x,g,group,64).cpu()
            if not torch.equal(out,ref):
                raise ValueError(f"mismatch at {(b,l,group)}")
        for order in (groups,groups[::-1],groups,groups[::-1]):
            for group in order:
                torch.npu.synchronize()
                start=time.perf_counter_ns()
                for _ in range(args.repeats):
                    evaluate(x,g,group,64)
                torch.npu.synchronize()
                rows.append(dict(batch=b,length=l,group=group,order=list(order),
                    ms=(time.perf_counter_ns()-start)/1e6/args.repeats))
    medians=[dict(batch=b,length=l,group=k,median_ms=median(r["ms"] for r in rows
             if r["batch"]==b and r["length"]==l and r["group"]==k))
             for b,l in ((1,19),(2,14),(2,22),(4,31)) for k in groups]
    args.output.write_text(json.dumps(dict(passed=True,runs=rows,medians=medians),indent=2))


if __name__=="__main__":
    main()
