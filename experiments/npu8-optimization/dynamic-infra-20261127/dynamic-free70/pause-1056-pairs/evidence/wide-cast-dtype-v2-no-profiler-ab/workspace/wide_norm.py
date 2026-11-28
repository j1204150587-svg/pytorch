"""Cast the published gradient buffer once before ZeRO-2 row-wise L2 norms."""
import atexit
import json
import os
from pathlib import Path

import torch


def join_adjacent_batches(plan):
    result = []
    for offset, shape, count, width in plan:
        if result:
            first, last_shape, n, last_width = result[-1]
            if (count > 1 and n > 1 and shape == last_shape and width == last_width
                    and first + n*width == offset):
                result[-1] = (first, shape, n+count, width)
                continue
        result.append((offset, shape, count, width))
    return result


def install(rank_dir):
    import flat_stats
    mode = os.environ.get("NPU_DYNAMIC_WIDE_NORM", "off")
    if mode not in ("off","cast","join"):
        raise ValueError(mode)
    seen = dict(mode=mode, calls=0, fallback=0, wide_casts=0, original_groups=0,
                submitted_norm_groups=0, max_wide_bytes=0)
    def save():
        (Path(rank_dir)/"wide_norm_counts.json").write_text(json.dumps(seen, indent=2)+"\n")
    atexit.register(save)
    if mode=="off":
        return
    original = flat_stats.row_norm_square
    def wide(flat, plan):
        seen["calls"] += 1
        if (flat.ndim!=1 or flat.dtype!=torch.bfloat16 or not flat.is_contiguous() or flat.requires_grad
                or not plan or plan[-1][0] + plan[-1][2]*plan[-1][3] != flat.numel()):
            seen["fallback"] += 1
            return original(flat,plan)
        with torch.profiler.record_function("PTAWideGradientNorm"):
            full = flat.double()
            groups = join_adjacent_batches(plan) if mode=="join" else plan
            # The baseline's per-slice .double() now returns the slice unchanged.
            # Its singleton and multi-row reductions, square and final sum still execute.
            result = original(full, groups)
        seen["wide_casts"] += 1
        seen["original_groups"] += len(plan)
        seen["submitted_norm_groups"] += len(groups)
        seen["max_wide_bytes"] = max(seen["max_wide_bytes"],full.numel()*full.element_size())
        return result
    flat_stats.row_norm_square = wide
