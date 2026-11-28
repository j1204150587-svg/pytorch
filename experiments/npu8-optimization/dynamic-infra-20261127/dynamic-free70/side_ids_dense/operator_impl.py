"""Compare original eager preparation with the identical dense transform under two launchers."""
from collections import Counter
import os

import torch
import torch_npu

from kernels import dense_side_ids
from side_ids_driver import active_hooks
from side_ids_impl import make_plan, reason, reference


counts = Counter()
plans = {}
witnesses = {}


def dense(ids, offsets, route):
    temp = torch.empty_like(ids, dtype=torch.int64, memory_format=torch.contiguous_format)
    grid = (ids.shape[0] * ids.shape[1], 2, 1)
    stream = torch.npu.current_stream(ids.device)
    for tensor in (ids, offsets, temp):
        tensor.record_stream(stream)
    def ordinary():
        return dense_side_ids[grid](ids, offsets, temp)
    if route == "S":
        ordinary()
        counts["triton"] += 1
        return temp
    if route != "P":
        raise ValueError(route)
    if os.getenv("TORCHINDUCTOR_NPU_FAST_LAUNCH") != "1" or torch.autograd.profiler._is_profiler_enabled:
        ordinary()
        counts["fallback_switch_or_profiler"] += 1
        return temp
    key = (ids.device.index, ids.dtype)
    saved = plans.get(key)
    if saved is None:
        binary = ordinary()
        try:
            plan, cap, evidence = make_plan(binary, dense_side_ids, ids)
        except (AttributeError, RuntimeError, ValueError) as exc:
            plan, cap, evidence = None, None, {"reason": str(exc)}
        plans[key] = binary, plan, cap
        witnesses[str(key)] = evidence
        counts["cold"] += 1
        return temp
    binary, plan, cap = saved
    raw = torch_npu._C._npu_getCurrentRawStreamNoWait(ids.device.index)
    if plan is None or not raw or active_hooks(binary):
        ordinary()
        counts["fallback_plan_stream_hooks"] += 1
        return temp
    counts["entered"] += 1
    cap(plan, int(raw), *grid, (ids, offsets, temp), 40)
    counts["returned"] += 1
    return temp


def prepare(ids, offsets, rows, route):
    if route == "A":
        counts["stock"] += 1
        return reference(ids, offsets, rows)
    why = reason(ids, offsets, rows)
    if why is not None:
        counts["fallback_" + why] += 1
        return reference(ids, offsets, rows)
    return dense(ids, offsets, route).movedim(2, 0).contiguous()
