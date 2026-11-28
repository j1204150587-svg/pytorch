"""Dynamic BF16 RMS backward. Both reductions and the cubic power remain stock."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_npu
from kernels import rms_grad_products, rms_grad_variance


def fused(args, launchers, native):
    x, w, inv, z, dy = args
    rows = x.numel() // 64
    if (x.ndim != 3 or x.size(-1) != 64 or w.shape != (64,) or dy.shape != x.shape
            or z.shape != x.shape or inv.shape != (*x.shape[:-1], 1)
            or any(v.dtype != torch.bfloat16 or v.device != x.device or not v.is_contiguous() for v in args)):
        return torch.ops.npu_hstu_rms_lut.backward_parts.default(*args)
    direct, row_products, weight_products = (torch.empty_like(x) for _ in range(3))
    launchers[0]((x, w, inv, z, dy, direct, row_products, weight_products), rows, native)
    dw = weight_products.sum_to_size(w.size())
    dr = row_products.sum_to_size(inv.size())
    cube = inv.pow(3)
    dx = torch.empty_like(x)
    launchers[1]((x, dr, cube, dx), rows, native)
    return direct, dx, dw


def diff(x, y):
    a = x.cpu().contiguous().view(torch.int16)
    b = y.cpu().contiguous().view(torch.int16)
    changed = (a != b).reshape(-1)
    indices = changed.nonzero().flatten()
    different = indices.numel()
    result = {"different": different, "elements": changed.numel()}
    if different:
        i = indices[0].item()
        result["first"] = [i, int(a.flatten()[i]) & 65535, int(b.flatten()[i]) & 65535]
        both_nan = a.view(torch.bfloat16).isnan() & b.view(torch.bfloat16).isnan()
        result["different_excluding_nan_payloads"] = (changed & ~both_nan.flatten()).sum().item()
    return result


def inputs(batch, length, generator, case):
    x, z, dy = [torch.randn(batch, length, 64, generator=generator).bfloat16() for _ in range(3)]
    w = torch.randn(64, generator=generator).bfloat16()
    inv = torch.rand(batch, length, 1, generator=generator).bfloat16() * 2
    if case == "limits":
        raw = [0x0000, 0x8000, 0x0001, 0x8001, 0x0080, 0x7f7f, 0xbf80, 0x3f80]
        dy[0, 0, :len(raw)] = torch.tensor(raw, dtype=torch.int32).short().view(torch.bfloat16)
    if case == "nonfinite":
        dy[0, 0, :3] = torch.tensor([float("nan"), float("inf"), -float("inf")]).bfloat16()
    return tuple(v.npu() for v in (x, w, inv, z, dy))


def run(cli, report):
    sys.path.insert(0, str(cli.fold_dir))
    from plan import Launcher
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.ops.load_library(str(cli.library))
    launchers = (Launcher(rms_grad_products, 8), Launcher(rms_grad_variance, 4))
    reference = torch.ops.npu_hstu_rms_lut.backward_parts.default
    shapes = json.loads(cli.schedule.read_bytes())
    focus = ([1, 19], [2, 14], [2, 22], [4, 31])
    gen = torch.Generator().manual_seed(1234)
    streams = [torch.npu.current_stream(), torch.npu.Stream()]
    report["cases"] = []
    examples = []
    for i, (b, l) in enumerate(shapes):
        cases = ("finite", "limits", "nonfinite") if [b, l] in focus else ("finite",)
        for case in cases:
            with torch.npu.stream(streams[i % 2]):
                tensors = inputs(b, l, gen, case)
                if [b, l] in focus and case == "finite":
                    examples.append(tensors)
                expected = reference(*tensors)
                actual = fused(tensors, launchers, True)
                checks = [diff(a, v) for a, v in zip(expected, actual, strict=True)]
                report["cases"].append(dict(shape=[b, l], case=case, stream=i % 2, edges=checks))
    torch.npu.synchronize()
    report["plans"] = [op.evidence for op in launchers]
    report["gate_counts"] = [dict(op.calls) for op in launchers]
    report["exact"] = all(edge["different"] == 0 for row in report["cases"] for edge in row["edges"])
    if not report["exact"]:
        return
    report["windows"] = []
    for order in ("NSP", "PSN", "NSP", "PSN"):
        row = {}
        for mode in order:
            torch.npu.synchronize()
            before = [Counter(op.calls) for op in launchers]
            host = []
            start = time.perf_counter_ns()
            for i in range(64):
                tensors = examples[i % len(examples)]
                begin = time.perf_counter_ns()
                out = reference(*tensors) if mode == "N" else fused(tensors, launchers, mode == "P")
                host.append((time.perf_counter_ns() - begin) / 1000)
            torch.npu.synchronize()
            elapsed = (time.perf_counter_ns() - start) / 64000
            row[mode] = dict(sync_us=elapsed, host_p50=statistics.median(host), host_us=host,
                             counters=[dict(op.calls - a) for op, a in zip(launchers, before)])
        report["windows"].append(dict(order=order, modes=row))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for flag in ("library", "schedule", "fold-dir", "output"):
        cli.add_argument("--" + flag, type=Path, required=True)
    args = cli.parse_args()
    result = {}
    try:
        run(args, result)
    finally:
        args.output.write_text(json.dumps(result, indent=2))
