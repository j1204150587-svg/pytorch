import argparse
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import torch
from norm_by_shape import ParamNormGroups


def main(args):
    sys.path.insert(0, str(args.workspace))
    sys.path.insert(0, str(args.wide_dir))
    from model import MultiTableHSTU, ModelConfig
    from flat_stats import norm_plan, row_norm_square
    from wide_norm import join_adjacent_batches
    if args.device == "npu":
        import torch_npu
        torch.npu.set_device(0)
    torch.set_num_threads(2)
    config = json.loads((args.workspace / "configs/model_many_tables.json").read_bytes())
    model = MultiTableHSTU(ModelConfig(**config))
    parameters = list(model.parameters())
    stage = SimpleNamespace(PIPE_REPLICATED="pipeline_replicated", is_model_parallel_parameter=lambda p: False)
    original = norm_plan(parameters, stage, SimpleNamespace(model_parallel_rank=0), 256 * 1024, 64)
    joined = join_adjacent_batches(original)
    candidate = ParamNormGroups(original, args.device)
    gen = torch.Generator().manual_seed(1234)
    cases = []
    flat = torch.randn(candidate.total, generator=gen).bfloat16()
    for case in ("normal", "sparse", "wide", "near-clip", "small", "zeros", "inf-first", "nan-last"):
        data = flat.clone()
        if case == "sparse":
            data[1:] = 0
        elif case == "wide":
            data[::151] *= 2 ** 15
        elif case == "near-clip":
            data /= data.numel() ** 0.5
        elif case == "small":
            data *= 2 ** -64
        elif case == "zeros":
            data.zero_()
        elif case == "inf-first":
            data[0] = float("inf")
        elif case == "nan-last":
            data[-1] = float("nan")
        full = data.to(args.device).double()
        a, b = row_norm_square(full, joined).cpu(), candidate(full).cpu()
        abits, bbits = int(a.view(torch.int32)), int(b.view(torch.int32))
        cases.append(dict(case=case, a=abits, b=bbits, exact=abits == bbits))
    report = dict(device=args.device, effective_dtype=str(full.dtype), wide_bytes=full.numel()*full.element_size(),
                  cases=cases, exact=all(r["exact"] for r in cases), groups=len(candidate.layout),
                  baseline_groups=len(joined), total=candidate.total, gather_elements=candidate.gather_elements,
                  index_bytes=(candidate.gather_elements + candidate.order.numel()) * 8, windows=[])
    if report["exact"] and args.device == "npu":
        inputs = [flat.npu(), flat.roll(1).npu(), flat.roll(2).npu(), flat.roll(3).npu()]
        for order in ("AB", "BA", "AB", "BA"):
            row = {"order": order}
            for leg in order:
                def call(i):
                    full = inputs[i % 4].double()
                    return row_norm_square(full, joined) if leg == "A" else candidate(full)
                for i in range(4):
                    call(i)
                torch.npu.synchronize()
                begin = time.perf_counter_ns()
                for i in range(32):
                    call(i)
                torch.npu.synchronize()
                row[leg] = (time.perf_counter_ns() - begin) / 32000
            report["windows"].append(row)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--workspace", type=Path, required=True)
    p.add_argument("--wide-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "npu"), required=True)
    p.add_argument("--output", type=Path, required=True)
    main(p.parse_args())
