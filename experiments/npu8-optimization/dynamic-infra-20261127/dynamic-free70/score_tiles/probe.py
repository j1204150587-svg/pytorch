"""Check full attention outputs and all three input gradients before synchronized timings."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import torch
import torch_npu


def run(args, report):
    sys.path.extend([str(args.fold_dir), str(args.attention_dir)])
    from micro import bits, build, execute, original
    import operator_impl as impl
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    def graph(fn):
        return lambda q, k, v: fn(q @ k.transpose(-2, -1)) @ v
    ref = original
    jit = graph(lambda x: impl.apply(x, False))
    plan = graph(lambda x: impl.apply(x, True))
    samples = []
    report["cases"] = []
    streams = (torch.npu.current_stream(), torch.npu.Stream())
    routes = ("output", "dQ", "dK", "dV")
    for i, (b, l) in enumerate(json.loads(args.schedule.read_bytes())):
        tags = ("finite", "signed_zero", "nonfinite") if i in (0, 1, 3) else ("finite",)
        for tag in tags:
            with torch.npu.stream(streams[i % 2]):
                inputs = build(b, l, "npu", tag, 250 + i)
                standard = execute(ref, inputs)
                ordinary = execute(jit, inputs)
                compiled = execute(plan, inputs)
                record = {"shape":[b, l], "case":tag, "stream":i % 2,
                          "edges":{k:bits(a,v) for k,a,v in zip(routes,standard,compiled,strict=True)},
                          "ordinary_edges":{k:bits(a,v) for k,a,v in zip(routes,standard,ordinary,strict=True)}}
                if i in (0, 1, 3, 4, 5, 6):
                    q, k, _, _ = inputs
                    raw = (q @ k.transpose(-2, -1)).detach().requires_grad_()
                    a, v = impl.stock(raw), impl.apply(raw, True)
                    upstream = torch.linspace(-2., 2., a.numel(), device="cpu").view(a.shape).bfloat16().to(raw.device)
                    da, = torch.autograd.grad(a, raw, upstream)
                    dv, = torch.autograd.grad(v, raw, upstream)
                    record["score_edges"] = {"scores":bits(a,v), "dScores":bits(da,dv)}
                report["cases"].append(record)
                if tag == "finite" and [b,l] in ([1,19],[2,14],[2,22],[4,31]):
                    samples.append(inputs)
    torch.npu.synchronize()
    report["plans"] = impl.EVIDENCE
    report["counts"] = dict(impl.COUNTS)
    report["exact"] = all(d["differing_bf16_words"] == 0 for rec in report["cases"]
                          for group in ("edges", "ordinary_edges", "score_edges") for d in rec.get(group,{}).values())
    report["windows"] = []
    if not report["exact"]:
        return
    for order in ("NSP", "PSN", "NSP", "PSN"):
        window = {"order":order}
        for route in order:
            fn = {"N":ref,"S":jit,"P":plan}[route]
            for sample in samples:
                execute(fn, sample)
            torch.npu.synchronize()
            before = Counter(impl.COUNTS)
            start = time.perf_counter_ns()
            for i in range(48):
                y = execute(fn, samples[i % len(samples)])
            torch.npu.synchronize()
            window[route] = {"us":(time.perf_counter_ns()-start)/48000,
                             "calls":dict(impl.COUNTS - before)}
        report["windows"].append(window)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("schedule", "fold-dir", "attention-dir", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    a = p.parse_args()
    data = {}
    try:
        run(a,data)
    finally:
        a.output.write_text(json.dumps(data,indent=2))
