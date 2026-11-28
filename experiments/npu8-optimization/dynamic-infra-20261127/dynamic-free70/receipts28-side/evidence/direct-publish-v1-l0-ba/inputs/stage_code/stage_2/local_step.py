"""Reuse ZeRO-2 flat storage within the single-rank HSTU experiment."""

import hashlib
import inspect
import json
from pathlib import Path
import textwrap
from types import MethodType


STEP_HASH = "44750b4ab49af6bb844cb020d3c2306f474a32c90ff4e7d56b140b75187dc966"
FLATTEN_HASH = "09da0d97d00d19025d07da033eba86f4c3af84ab15845ed41f0008f3ab6fecd1"
ALIASES_HASH = "a682aa3890a3e3f0e3991347bf41aad800c4a31195d332124b2b0a0198da3406"


def _verify_method(optimizer, name, expected):
    method = getattr(optimizer, name)
    source = textwrap.dedent(inspect.getsource(method)).strip()
    if name in vars(optimizer) or hashlib.sha256(source.encode()).hexdigest() != expected:
        raise RuntimeError(f"unexpected DeepSpeed implementation of {name}")
    return method


def _check_weight_views(optimizer):
    weights, = optimizer.bit16_groups
    reordered, = optimizer.round_robin_bit16_groups
    indices, = optimizer.round_robin_bit16_indices
    flat, = optimizer.bit16_groups_flat
    partitions, = optimizer.parallel_partitioned_bit16_groups
    partition, = partitions
    pointer = flat.data_ptr()
    if (partition.data_ptr() != pointer or partition.shape != flat.shape
            or partition.dtype != flat.dtype or partition.device != flat.device
            or not partition.is_contiguous() or not flat.is_contiguous()):
        raise RuntimeError("single-rank weight partition must alias the complete BF16 buffer")

    offset = 0
    for index, view in enumerate(reordered):
        if (view.data_ptr() != pointer + offset * flat.element_size()
                or view.dtype != flat.dtype or view.device != flat.device or not view.is_contiguous()):
            raise RuntimeError(f"unexpected BF16 storage for reordered weight {index}")
        offset += view.numel()
    if offset != flat.numel() or len(weights) != len(reordered):
        raise RuntimeError("BF16 views do not cover the complete flat buffer")
    for index, weight in enumerate(weights):
        view = reordered[indices[index]]
        if (weight.data_ptr() != view.data_ptr() or weight.shape != view.shape
                or weight.stride() != view.stride() or weight.dtype != view.dtype or weight.device != view.device):
            raise RuntimeError(f"live BF16 weight {index} does not alias its flat view")
    return flat, partition, pointer


def install(optimizer, device, mode, rank_dir):
    _verify_method(optimizer, "step", STEP_HASH)
    if (mode not in ("flatten", "aliases", "both") or len(optimizer.params_in_partition) != 1
            or optimizer.groups_padding[0] or optimizer.cpu_offload or optimizer.round_robin_gradients
            or not hasattr(optimizer, "_recsys_local_flat_gradients")):
        raise RuntimeError("local optimizer path requires the unpadded single-rank gradient partition")
    counts = dict(mode=mode, flatten=0, aliases=0)

    if mode in ("flatten", "both"):
        original_flatten = _verify_method(optimizer, "flatten_dense_tensors_aligned", FLATTEN_HASH)

        def flatten(self, tensors, alignment, use_cpu_data=False):
            current = self._recsys_local_flat_gradients.get(0)
            if use_cpu_data or current is None or tensors is not current[0]:
                return original_flatten(tensors, alignment, use_cpu_data)
            buffer = current[1]
            if (alignment != buffer.numel() or buffer.ndim != 1 or buffer.dtype != self.dtype
                    or buffer.device != device or not buffer.is_contiguous()
                    or buffer.dtype == self.single_partition_of_fp32_groups[0].dtype):
                raise RuntimeError("flat gradient buffer cannot be reused before the FP32 cast")
            counts["flatten"] += 1
            return buffer

        optimizer.flatten_dense_tensors_aligned = MethodType(flatten, optimizer)

    if mode in ("aliases", "both"):
        original_update = _verify_method(optimizer, "_update_model_bit16_weights", ALIASES_HASH)
        flat, partition, pointer = _check_weight_views(optimizer)

        def reuse_weight_views(self, group_index):
            if group_index != 0:
                return original_update(group_index)
            if (self.bit16_groups_flat[0] is not flat or self.parallel_partitioned_bit16_groups[0][0] is not partition
                    or flat.data_ptr() != pointer or partition.data_ptr() != pointer):
                raise RuntimeError("BF16 storage moved after optimizer specialization")
            counts["aliases"] += 1

        optimizer._update_model_bit16_weights = MethodType(reuse_weight_views, optimizer)

    def finish():
        (Path(rank_dir) / "zero2_local_step.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")

    return finish
