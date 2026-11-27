"""Compare fused row statistics to the exact installed dynamic eager ATen reductions."""
import argparse
import json
from pathlib import Path
import time
import torch
import torch_npu
from rows import PartitionMoments


def main():
    import sys
    cli = argparse.ArgumentParser()
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--workspace", type=Path, required=True)
    args = cli.parse_args()
    sys.path.insert(0, str(args.workspace))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "flat_stats_stage"))
    from model import ModelConfig, MultiTableHSTU
    from flat_stats import row_norm_square, nonfinite, norm_plan
    from zero2_batch_norm import CHUNK_ELEMENTS, CHUNK_TENSORS
    from types import SimpleNamespace
    torch.manual_seed(1234)
    cfg = ModelConfig(**json.loads((args.workspace / "configs/model_many_tables.json").read_bytes()))
    params = list(MultiTableHSTU(cfg).parameters())
    sizes = [p.numel() for p in params]
    plan = norm_plan(params, SimpleNamespace(PIPE_REPLICATED="ds_pipe_replicated", is_model_parallel_parameter=lambda p:False),
                     SimpleNamespace(model_parallel_rank=0), CHUNK_ELEMENTS, CHUNK_TENSORS)
    torch.npu.set_device(0)
    candidate = PartitionMoments(sizes, torch.device("npu", 0))
    flat = torch.randn(sum(sizes), dtype=torch.float32).bfloat16().npu()
    rows = []
    for label, sentinel, index in (("random", 0., None), ("nan", float("nan"), 0), ("negative_inf", -float("inf"), 69696),
                                  ("positive_inf", float("inf"), -1), ("max_finite", torch.finfo(torch.bfloat16).max, -1)):
        sample = flat.clone()
        if index is not None:
            sample[index].fill_(sentinel)
        overflow = bool(candidate.scan(sample))
        ref_overflow = bool(nonfinite(sample))
        actual, reference = candidate.square(sample), row_norm_square(sample, plan)
        same = bool(torch.allclose(actual, reference, rtol=1e-6, atol=1e-6, equal_nan=True))
        rows.append(dict(case=label, overflow_equal=overflow==ref_overflow, norm_close=same,
                         exact_finite_square=bool(torch.equal(actual,reference)) if torch.isfinite(reference) else None))
    times=[]
    def baseline():
        return nonfinite(flat), row_norm_square(flat, plan)
    def treatment():
        return candidate.scan(flat), candidate.square(flat)
    for order in ("AB", "BA"):
        for v in order:
            fn = baseline if v=="A" else treatment
            for _ in range(3):
                fn()
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(24):
                fn()
            torch.npu.synchronize()
            times.append(dict(order=order, variant=v, microseconds=(time.perf_counter()-start)*1e6/24))
    passed = all(r["overflow_equal"] and r["norm_close"] for r in rows)
    args.output.write_text(json.dumps(dict(passed=passed, parameters=len(params), elements=sum(sizes), cases=rows, times=times), indent=2))
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
