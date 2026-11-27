"""Opt-in replacement for the local BF16 ZeRO-2 overflow scan only."""

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import inspect
import json
import os
import sys
import textwrap
from types import MethodType

import torch


SWITCH = "RECSYS_ZERO2_BATCH_OVERFLOW"
CHUNK_ELEMENTS = 256 * 1024
ACL_FORMAT_ND = 2
PINNED_METHODS = {
    "step": "44750b4ab49af6bb844cb020d3c2306f474a32c90ff4e7d56b140b75187dc966",
    "has_overflow": "7bb9d1f516f05fc446ab6aa6755f7994e252a3d15db8859abc56a7c9d70366ff",
    "has_overflow_partitioned_grads_serial": "b54b01f27a954af8eaabbbf5af3ba8a7aad2c631217a87a474e53c4f34064e49",
    "_has_inf_or_nan": "d3252aef4efb2f28aa999d5834c2e65d397cc39f6c1e8fe585f5d1275f5971cd",
}


def enabled():
    value = os.environ.get(SWITCH, "0")
    if value not in ("0", "1"):
        raise ValueError(f"{SWITCH} must be 0 or 1, got {value!r}")
    return value == "1"


def emit(event, **details):
    print(json.dumps({"component": "zero2_batch_overflow", "event": event, **details},
                     sort_keys=True), file=sys.stderr, flush=True)


def shard_metadata(groups):
    counts = Counter()
    for group in groups:
        for grad in group:
            if grad is None:
                counts[("None", "-", "-")] += 1
            elif isinstance(grad, torch.Tensor):
                counts[(str(grad.dtype), str(grad.device), str(grad.layout))] += 1
            else:
                counts[(type(grad).__name__, "-", "-")] += 1
    return [{"dtype": dtype, "device": device, "layout": layout, "count": count}
            for (dtype, device, layout), count in sorted(counts.items())]


@dataclass
class ScanStats:
    calls: int = 0
    fast_calls: int = 0
    fallback_calls: int = 0
    errors: int = 0
    chunks: int = 0
    reasons: Counter = field(default_factory=Counter)
    first_shards: list | None = None
    first_npu_formats: dict | None = None

    def snapshot(self):
        return {
            "calls": self.calls, "fast_calls": self.fast_calls,
            "fallback_calls": self.fallback_calls, "errors": self.errors,
            "chunks": self.chunks, "reasons": dict(self.reasons),
            "first_shards": self.first_shards,
            "first_npu_formats": self.first_npu_formats,
        }


def chunk_plan(groups, device, chunk_elements, storage_format=None, formats=None):
    chunks, pending = [], []
    pending_size = 0
    for group in groups:
        if type(group) not in (list, tuple):
            return None, "partition_not_list"
        for grad in group:
            if grad is None:
                continue
            if type(grad) is not torch.Tensor:
                return None, "not_plain_tensor"
            if grad.dtype != torch.bfloat16:
                return None, "not_bf16"
            if grad.device != device:
                return None, "device_mismatch"
            if (grad.layout != torch.strided or not grad.is_contiguous()
                    or grad.is_neg() or grad.is_conj() or grad.requires_grad):
                return None, "unsupported_layout_or_view"
            if storage_format is not None:
                format_id = storage_format(grad)
                if formats is not None:
                    formats[format_id] += 1
                if format_id != ACL_FORMAT_ND:
                    return None, "unsupported_npu_storage_format"
            size = grad.numel()
            if size == 0:
                return None, "empty_tensor"
            if size > chunk_elements:
                return None, "tensor_exceeds_chunk"
            if pending_size + size > chunk_elements:
                chunks.append(pending)
                pending, pending_size = [], 0
            pending.append(grad)
            pending_size += size
    if pending:
        chunks.append(pending)
    if not chunks:
        return None, "no_gradients"
    return chunks, None


def batched_scan(groups, device, scan_tensor, legacy, stats, *, chunk_elements=CHUNK_ELEMENTS,
                 storage_format=None):
    stats.calls += 1
    path = "plan"
    try:
        if stats.first_shards is None and all(type(group) in (list, tuple) for group in groups):
            stats.first_shards = shard_metadata(groups)
        formats = Counter() if storage_format is not None and stats.first_npu_formats is None else None
        chunks, reason = chunk_plan(groups, device, chunk_elements, storage_format, formats)
        if formats is not None:
            stats.first_npu_formats = dict(formats)
        if reason is not None:
            path = "legacy_fallback"
            stats.fallback_calls += 1
            stats.reasons[reason] += 1
            if stats.reasons[reason] == 1:
                emit("fallback", reason=reason, scan=stats.snapshot())
            return legacy()

        path = "batched"
        invalid_count = torch.zeros([1], dtype=torch.float, device=device)
        for chunk in chunks:
            data = chunk[0] if len(chunk) == 1 else torch.cat([grad.view(-1) for grad in chunk])
            invalid_count += scan_tensor(data)
        result = invalid_count.bool()
        stats.fast_calls += 1
        stats.chunks += len(chunks)
        return result
    except Exception as exc:
        stats.errors += 1
        emit("scan_error", path=path, error_type=type(exc).__name__, error=str(exc),
             scan=stats.snapshot())
        raise


def install(optimizer):
    if not enabled():
        return None
    try:
        return _install(optimizer)
    except Exception as exc:
        emit("install_error", error_type=type(exc).__name__, error=str(exc))
        raise


def _install(optimizer):
    import deepspeed
    from deepspeed import comm as dist
    from deepspeed.accelerator import get_accelerator
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    if deepspeed.__version__ != "0.19.0" or type(optimizer) is not DeepSpeedZeroOptimizer:
        raise RuntimeError("candidate requires DeepSpeed 0.19.0 DeepSpeedZeroOptimizer")
    if "_recsys_overflow_stats" in vars(optimizer):
        raise RuntimeError("candidate already installed")
    actual = {}
    for name, expected in PINNED_METHODS.items():
        method = getattr(optimizer, name)
        if name in vars(optimizer):
            raise RuntimeError(f"candidate refuses a pre-existing instance override: {name}")
        source = textwrap.dedent(inspect.getsource(method)).strip()
        actual[name] = hashlib.sha256(source.encode()).hexdigest()
        if actual[name] != expected:
            raise RuntimeError(f"DeepSpeed source mismatch for {name}: {actual[name]} != {expected}")

    device = torch.device(get_accelerator().current_device_name())
    compatible = (device.type == "npu" and device.index is not None and dist.get_world_size() == 1
                  and optimizer.partition_gradients and optimizer.check_grad_overflow
                  and not optimizer.cpu_offload and optimizer.dtype == torch.bfloat16
                  and optimizer.gradient_accumulation_dtype == torch.bfloat16
                  and optimizer.model_parallel_group is None and not optimizer.has_moe_layers
                  and not optimizer.zenflow and optimizer.torch_autocast_gradscaler is None)
    if not compatible:
        raise RuntimeError("candidate requires single-NPU BF16 ZeRO-2 with overflow checking, "
                           "BF16 accumulation, and no offload, MP, MoE, ZenFlow or AMP scaler; "
                           f"device={device}, dtype={optimizer.dtype}, "
                           f"grad_accum_dtype={optimizer.gradient_accumulation_dtype}")

    legacy = optimizer.has_overflow_partitioned_grads_serial
    stats = ScanStats()
    storage_format = getattr(torch.ops.npu, "get_npu_format", None)
    if storage_format is None:
        raise RuntimeError("torch.ops.npu.get_npu_format is required for NPU layout validation")

    def local_scan(self):
        groups = [self.averaged_gradients[index] for index in range(len(self.bit16_groups))]
        current_device = torch.device(get_accelerator().current_device_name())
        return batched_scan(groups, current_device, self._has_inf_or_nan, legacy, stats,
                            storage_format=storage_format)

    optimizer.has_overflow_partitioned_grads_serial = MethodType(local_scan, optimizer)
    optimizer._recsys_overflow_stats = stats
    emit("installed", device=str(device), grad_accum_dtype=str(optimizer.gradient_accumulation_dtype),
         chunk_elements=CHUNK_ELEMENTS, deepspeed=deepspeed.__version__, source_hashes=actual)
    return stats


def finish_run(stats, rank_dir, expected_steps, training_completed):
    snapshot = stats.snapshot()
    valid = (training_completed and expected_steps > 0
             and snapshot["calls"] == snapshot["fast_calls"] == expected_steps
             and not snapshot["fallback_calls"] and not snapshot["errors"])
    report = {"training_completed": training_completed, "valid": valid,
              "expected_steps": expected_steps, **snapshot}
    (rank_dir / "overflow_scan.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    emit("run_scan_summary", **report)
    if training_completed and not valid:
        raise RuntimeError("invalid candidate run: exactly one fast scan per step and zero fallbacks/errors required")
