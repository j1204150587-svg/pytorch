"""Gate one attention candidate on real projection strides, then time A/B and B/A."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter_ns
import traceback

import torch
import torch.nn.functional as F

from attention_case import full, layout, original, partial


SHAPES = [[1,19], [2,14], [2,22], [1,12], [3,16], [4,31], [4,32], [1,25]]


def build(b, l, device, tag, seed):
    rng = torch.Generator().manual_seed(seed)
    proj_input = torch.randn(b,l,64,generator=rng).bfloat16().to(device)
    proj_weight = (torch.randn(256,64,generator=rng) * (64 ** -.5)).bfloat16().to(device)
    projection = F.linear(proj_input, proj_weight)
    upstream = torch.randn(b,l,4,16,generator=rng).bfloat16()
    if tag == "signed_zero":
        projection[..., 128:144] = -0.
        projection[..., 64:80] = 0.
        upstream[:,::2] = -0.
    elif tag == "nonfinite":
        projection[..., 0, 128] = float("nan")
        projection[..., l-1, 64+16] = float("inf")
        upstream[:,l//2,2,2] = float("-inf")
    elif tag != "finite":
        raise ValueError(tag)
    _, v, q, k = [p.view(b,l,4,16).transpose(1,2).detach().requires_grad_(True)
                  for p in projection.split(64, -1)]
    dy = upstream.to(device).transpose(1,2)
    if not layout(q,k,v):
        raise RuntimeError(f"unsupported projection layout {q.shape} {q.stride()}")
    return q,k,v,dy


def execute(fn, inputs):
    q,k,v,dy = inputs
    y = fn(q,k,v)
    return (y.detach(), *torch.autograd.grad(y, (q,k,v), dy))


def bits(left, right):
    a = left.detach().cpu().contiguous()
    b = right.detach().cpu().contiguous()
    aa, bb = (x.view(torch.int16) for x in (a,b))
    bad = aa != bb
    count = int(bad.sum())
    if not count:
        return {"differing_bf16_words":0}
    pos = tuple(bad.nonzero()[0].tolist())
    finite = a.isfinite() & b.isfinite()
    return {"differing_bf16_words":count, "first_index":pos, "A_word":int(aa[pos]) & 65535,
            "B_word":int(bb[pos]) & 65535, "different_finite_values":int(((a != b) & finite).sum()),
            "first_numerical_absolute_max":float((a[finite].float()-b[finite].float()).abs().max()) if finite.any() else None}


def save(report, path):
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(report,indent=2) + "\n",encoding="utf-8")
    tmp.replace(path)


def run(args):
    device = args.device
    if device == "npu":
        import torch_npu
        torch.npu.set_device(0)
        sync = torch.npu.synchronize
    else:
        torch.set_num_threads(1)
        sync = lambda: None
    shapes = json.loads(args.schedule.read_text(encoding="utf-8")) if args.schedule else SHAPES
    fn = {"full":full, "partial":partial}[args.mode]
    report = {"candidate":args.mode,"device":device,"torch":torch.__version__,"platform":platform.platform(),
              "schedule":shapes,"cases":[],"correct":False,"timings":[],"complete":False,
              "sources":{s:hashlib.sha256(Path(__file__).with_name(s).read_bytes()).hexdigest()
                         for s in ("micro.py","attention_case.py","attn_kernels.py")}}
    finite_inputs = []
    try:
        for idx,(b,l) in enumerate(shapes):
            tags = ["finite"]
            if idx in (0,1,3):
                tags += ["signed_zero","nonfinite"]
            for kind in tags:
                seed = 250 + idx
                inputs = build(b,l,device,kind,seed)
                q,k,v,g = inputs
                result = {"shape":[b,l], "case":kind, "seed":seed, "strides":[list(x.stride()) for x in inputs],
                          "storage_offsets_QKV":[x.storage_offset() for x in (q,k,v)],
                          "same_projection_storage": len({x.untyped_storage()._cdata for x in (q,k,v)}) == 1}
                if device == "npu":
                    result["QKV_formats"] = [int(torch.ops.npu.get_npu_format(x)) for x in (q,k,v)]
                reference = execute(original,inputs)
                candidate = execute(fn,inputs)
                result["words"] = {key:bits(a,c) for key,a,c in zip(("Y","dQ","dK","dV"),reference,candidate)}
                result["passed"] = all(item["differing_bf16_words"] == 0 for item in result["words"].values())
                report["cases"].append(result)
                if kind == "finite":
                    finite_inputs.append(inputs)
                save(report,args.out)
        report["correct"] = all(c["passed"] for c in report["cases"])
        if report["correct"] and args.repeats:
            for order in ("AB","BA","AB","BA"):
                t = {}
                for leg in order:
                    op = original if leg == "A" else fn
                    for example in finite_inputs:
                        execute(op,example)
                    sync()
                    start = perf_counter_ns()
                    for i in range(args.repeats):
                        execute(op,finite_inputs[i % len(finite_inputs)])
                    sync()
                    t[leg] = (perf_counter_ns()-start)/1000/args.repeats
                report["timings"].append(dict(order=order,calls=args.repeats,us_per_forward_backward=t,
                                              pct_latency_reduction=(1-t["B"]/t["A"])*100))
                save(report,args.out)
        report["complete"] = True
    except Exception:
        report["error"] = traceback.format_exc()
    save(report,args.out)
    print(json.dumps({k:v for k,v in report.items() if k in ("correct","complete","error","timings")}),flush=True)
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--mode",choices=("full","partial"),required=True)
    cli.add_argument("--device",choices=("cpu","npu"),required=True)
    cli.add_argument("--out",type=Path,required=True)
    cli.add_argument("--schedule",type=Path)
    cli.add_argument("--repeats",type=int,default=48)
    sys.exit(run(cli.parse_args()))
