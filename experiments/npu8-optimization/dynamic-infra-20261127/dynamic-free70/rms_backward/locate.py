"""Locate the first divergent boundary and measure the product-only fast path."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
import torch_npu
from kernels import rms_grad_products, rms_grad_variance
from micro import diff, inputs


def prefix(t, launcher):
    x, w, inv, z, dy = t
    direct, row, weight = [torch.empty_like(x) for _ in range(3)]
    launcher((x, w, inv, z, dy, direct, row, weight), x.numel() // 64, True)
    return direct, row, weight


def main(args):
    sys.path.insert(0, str(args.fold_dir))
    from plan import Launcher
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    torch.ops.load_library(str(args.library))
    torch.ops.load_library(str(args.tail))
    products = Launcher(rms_grad_products, 8)
    variance = Launcher(rms_grad_variance, 4)
    reference = torch.ops.npu_hstu_rms_lut.backward_parts.default
    finish = torch.ops.npu_hstu_rms_grad_prefix.finish.default
    gen = torch.Generator().manual_seed(1234)
    data = dict(cases=[], windows=[])
    shapes = json.loads(args.schedule.read_bytes())
    examples = []
    for i, (b, l) in enumerate(shapes):
        for kind in (("finite", "limits", "nonfinite") if i < 4 else ("finite",)):
            t = inputs(b, l, gen, kind)
            x, w, inv, z, dy = t
            expected = reference(*t)
            direct, row, weight = prefix(t, products)
            edges = finish(x, w, inv, direct, row, weight)
            case = dict(shape=[b, l], case=kind,
                        prefix_only=[diff(a, q) for a, q in zip(expected, edges)])
            if i < 4:
                gz = dy * w
                stock_row = gz * x
                dr = stock_row.sum_to_size(inv.shape)
                cube = inv.pow(3)
                raw_dx = torch.empty_like(x)
                variance((x, dr, cube, raw_dx), b * l, True)
                case["stock_boundaries"] = dict(row_products=diff(stock_row, row),
                    weight_products=diff(dy * z, weight),
                    reduced_products=diff(dr, row.sum_to_size(inv.shape)),
                    native_manual_tail=diff(expected[1], (((dr * -0.5) * cube) / 64) * (x * 2)),
                    triton_tail_of_stock_inputs=diff(expected[1], raw_dx))
                if kind == "finite":
                    examples.append(t)
            data["cases"].append(case)
    data["prefix_exact"] = all(d["different"] == 0 for c in data["cases"] for d in c["prefix_only"])
    if data["prefix_exact"]:
        for order in ("AB", "BA", "AB", "BA"):
            timing = dict(order=order)
            for leg in order:
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                for i in range(64):
                    t = examples[i % len(examples)]
                    out = reference(*t) if leg == "A" else finish(*t[:3], *prefix(t, products))
                torch.npu.synchronize()
                timing[leg] = (time.perf_counter_ns() - start) / 64000
            data["windows"].append(timing)
    data["launches"] = dict(products.calls)
    args.output.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for flag in ("fold-dir", "library", "tail", "schedule", "output"):
        p.add_argument("--" + flag, type=Path, required=True)
    main(p.parse_args())
