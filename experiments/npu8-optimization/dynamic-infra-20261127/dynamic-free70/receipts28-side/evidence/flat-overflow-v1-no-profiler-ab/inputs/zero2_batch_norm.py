"""Opt-in batching of the per-gradient L2 norm in DeepSpeed 0.19.0 ZeRO-2."""

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
import textwrap
from types import MethodType

import torch


SWITCH = "RECSYS_ZERO2_BATCH_NORM"
CHUNK_ELEMENTS = 256 * 1024
CHUNK_TENSORS = 64
ACL_FORMAT_ND = 2
PINNED_METHODS = {
    "get_grad_norm_direct": "cad2fa322292007f163c56e9d1010b8fb6824d4d3efd7fbbe29695873b30bbdf",
    "step": "44750b4ab49af6bb844cb020d3c2306f474a32c90ff4e7d56b140b75187dc966",
}


@dataclass
class NormStats:
    calls: int = 0
    batched_calls: int = 0
    batches: int = 0
    batched_gradients: int = 0
    fallback_reasons: Counter = field(default_factory=Counter)

    def require_batched_step(self, completed_steps):
        if self.calls != completed_steps or self.batched_calls != completed_steps:
            raise RuntimeError(f"expected one batched norm per step through step {completed_steps}; "
                               f"calls={self.calls}, batched={self.batched_calls}, "
                               f"fallbacks={dict(self.fallback_reasons)}")

    def snapshot(self):
        return {
            "calls": self.calls,
            "batched_calls": self.batched_calls,
            "batches": self.batches,
            "batched_gradients": self.batched_gradients,
            "fallback_reasons": dict(self.fallback_reasons),
        }


def _plan_batches(optimizer, gradients, params, stage, device, storage_format,
                  chunk_elements, chunk_tensors):
    if type(gradients) not in (list, tuple) or type(params) not in (list, tuple):
        return None, "unsupported_sequence"
    batches = []
    batch = []
    batch_shape = None
    for grad, param in zip(gradients, params):
        if hasattr(param, stage.PIPE_REPLICATED) and param.ds_pipe_replicated:
            continue
        if not (stage.is_model_parallel_parameter(param) or optimizer.model_parallel_rank == 0):
            continue
        if type(grad) is not torch.Tensor or grad.dtype != torch.bfloat16:
            return None, "unsupported_dtype_or_tensor"
        if (grad.device != device or grad.layout != torch.strided or not grad.is_contiguous()
                or grad.is_conj() or grad.is_neg() or grad.requires_grad):
            return None, "unsupported_device_or_layout"
        if storage_format is not None and storage_format(grad) != ACL_FORMAT_ND:
            return None, "unsupported_npu_format"
        size = grad.numel()
        if size == 0:
            return None, "empty_gradient"
        if batch and (grad.shape != batch_shape or len(batch) >= chunk_tensors
                      or (len(batch) + 1) * size > chunk_elements):
            batches.append(batch)
            batch = []
        batch.append(grad)
        batch_shape = grad.shape
    if batch:
        batches.append(batch)
    return batches, None


def _group_norm(optimizer, gradients, params, norm_type, *, original, stage, device, stats,
                storage_format=None, chunk_elements=CHUNK_ELEMENTS, chunk_tensors=CHUNK_TENSORS):
    stats.calls += 1
    current_device = stage.get_accelerator().current_device_name()
    if type(norm_type) not in (int, float) or norm_type != 2:
        batches, reason = None, "unsupported_norm_type"
    elif torch.device(current_device) != device:
        batches, reason = None, "current_device_changed"
    else:
        batches, reason = _plan_batches(optimizer, gradients, params, stage, device, storage_format,
                                       chunk_elements, chunk_tensors)
    if reason is None and not any(len(batch) > 1 for batch in batches):
        reason = "no_batch"
    if reason is not None:
        stats.fallback_reasons[reason] += 1
        return original(gradients, params, norm_type=norm_type)

    norm_vectors = []
    packed_batches = packed_gradients = 0
    for batch in batches:
        if len(batch) == 1:
            norm = torch.linalg.vector_norm(batch[0].data.double().detach(), ord=2).reshape(1)
        else:
            packed = torch.stack(batch).view(len(batch), -1)
            # Request the same cast as DeepSpeed, including the backend's NPU behavior.
            norm = torch.linalg.vector_norm(packed.double().detach(), ord=2, dim=1)
            packed_batches += 1
            packed_gradients += len(batch)
        norm_vectors.append(norm.to(current_device))

    all_norms = norm_vectors[0] if len(norm_vectors) == 1 else torch.cat(norm_vectors)
    total_norm = all_norms.square().sum().float()
    dist = stage.dist
    dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=optimizer.dp_process_group)
    optimizer._model_parallel_all_reduce(tensor=total_norm, op=dist.ReduceOp.SUM)
    total_norm = total_norm.pow(0.5)
    stage.mask_nan_or_inf_with_val_inplace(total_norm, device=optimizer.device)
    stats.batched_calls += 1
    stats.batches += packed_batches
    stats.batched_gradients += packed_gradients
    return total_norm


def _verify_sources(optimizer):
    for name, expected in PINNED_METHODS.items():
        if name in vars(optimizer):
            raise RuntimeError(f"candidate refuses a pre-existing instance override: {name}")
        method = getattr(optimizer, name)
        source = textwrap.dedent(inspect.getsource(method)).strip()
        actual = hashlib.sha256(source.encode()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"DeepSpeed source mismatch for {name}: {actual} != {expected}")


def install(optimizer):
    """Install on the selected ZeRO instance before the existing profiler wrapper."""
    switch = os.environ.get(SWITCH, "0")
    if switch == "0":
        return None
    if switch != "1":
        raise ValueError(f"{SWITCH} must be 0 or 1, got {switch!r}")

    import deepspeed
    from deepspeed.runtime.zero import stage_1_and_2 as stage

    if deepspeed.__version__ != "0.19.0" or type(optimizer) is not stage.DeepSpeedZeroOptimizer:
        raise RuntimeError("candidate requires DeepSpeed 0.19.0 DeepSpeedZeroOptimizer")
    _verify_sources(optimizer)
    device = torch.device(stage.get_accelerator().current_device_name())
    if not (device.type == "npu" and device.index is not None
            and torch.device(optimizer.device) == device and stage.dist.get_world_size() == 1
            and optimizer.partition_gradients and optimizer.check_grad_overflow
            and not optimizer.cpu_offload and optimizer.dtype == torch.bfloat16
            and optimizer.gradient_accumulation_dtype == torch.bfloat16
            and optimizer.model_parallel_group is None and not optimizer.has_moe_layers
            and not optimizer.zenflow and optimizer.torch_autocast_gradscaler is None
            and optimizer.clip_grad > 0 and type(optimizer.optimizer) is torch.optim.AdamW):
        raise RuntimeError("candidate requires single-NPU BF16 ZeRO-2 with BF16 accumulation, "
                           "overflow checking, clipping and AdamW, and no offload, MP, MoE, "
                           "ZenFlow or AMP scaler")
    storage_format = getattr(torch.ops.npu, "get_npu_format", None)
    if storage_format is None:
        raise RuntimeError("torch.ops.npu.get_npu_format is required")

    original = optimizer.get_grad_norm_direct
    stats = NormStats()

    def get_grad_norm_direct(self, gradients, params, norm_type=2):
        return _group_norm(self, gradients, params, norm_type, original=original, stage=stage,
                           device=device, stats=stats, storage_format=storage_format)

    optimizer.get_grad_norm_direct = MethodType(get_grad_norm_direct, optimizer)
    return stats


def finish_run(stats, rank_dir, expected_steps, training_completed):
    snapshot = stats.snapshot()
    valid = (training_completed and expected_steps > 0
             and snapshot["calls"] == snapshot["batched_calls"] == expected_steps
             and not snapshot["fallback_reasons"])
    report = {"training_completed": training_completed, "valid": valid,
              "expected_steps": expected_steps, **snapshot}
    (rank_dir / "norm_stats.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if training_completed and not valid:
        raise RuntimeError("invalid norm candidate run; see norm_stats.json")
