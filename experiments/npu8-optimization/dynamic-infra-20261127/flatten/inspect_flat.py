"""Inspect the actual ZeRO flatten arguments and the #41 native scope on NPU."""
from collections import Counter
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys

import torch
import torch_npu

MARK = "NPUDenseFlattenStorageCopy"


def examine(tensors):
    tensors = list(tensors)
    first = tensors[0]
    storage = first.untyped_storage()
    initial = first.storage_offset()
    capacity = storage.nbytes() // first.element_size()
    end = initial
    failures = Counter()
    storage_ids = set()
    first_bad = None
    for i, tensor in enumerate(tensors):
        storage_id = tensor.untyped_storage()._cdata
        storage_ids.add(storage_id)
        checks = {
            "different_storage": storage_id != storage._cdata,
            "offset_not_adjacent": tensor.storage_offset() != end,
            "wrong_device_or_dtype": tensor.device != first.device or tensor.dtype != first.dtype,
            "requires_grad": tensor.requires_grad,
            "forward_tangent": torch.autograd.forward_ad.unpack_dual(tensor).tangent is not None,
            "conj_or_neg": tensor.is_conj() or tensor.is_neg(),
            "quantized_or_named": tensor.is_quantized or tensor.has_names(),
            "noncontiguous": not tensor.is_contiguous(),
            "empty": tensor.numel() == 0,
        }
        failed = [k for k, v in checks.items() if v]
        failures.update(failed)
        if failed and first_bad is None:
            first_bad = dict(index=i, causes=failed, offset=tensor.storage_offset(),
                             expected_offset=end, shape=list(tensor.shape), strides=list(tensor.stride()))
        end += tensor.numel()
    return dict(tensors=len(tensors), distinct_storages=len(storage_ids), elements=end-initial,
                first_shape=list(first.shape), dtype=str(first.dtype), device=str(first.device),
                first_npu_format=int(torch_npu.get_npu_format(first)),
                first_storage_capacity=capacity, range_fits=0 <= initial <= end <= capacity,
                disqualifying_metadata=failures, first_mismatch=first_bad)


def install(optimizer, rank_dir):
    library, = sorted({Path(row.split()[-1]) for row in Path("/proc/self/maps").read_text().splitlines()
                      if row.split()[-1].endswith("/libtorch_npu.so")})
    with library.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as image:
        marker_available = image.find(MARK.encode()) >= 0
        sha = hashlib.sha256(image).hexdigest()
    state = {"step": -1, "shape": None}
    output = Path(rank_dir) / "flatten-probe.jsonl"
    real_flatten = optimizer.flatten
    with output.open("w") as stream:
        print(json.dumps(dict(library=str(library), sha256=sha, marker_available=marker_available,
                              flat_switch=os.getenv("NPU_EXPERIMENT_FLAT_ND"),
                              callable_name=real_flatten.__qualname__)), file=stream)

    def observe(tensors, *args, **kwargs):
        row = examine(tensors)
        row.update(step=state["step"], shape=state["shape"], caller=sys._getframe(1).f_code.co_name)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            result = real_flatten(tensors, *args, **kwargs)
        names = Counter(event.name for event in prof.events())
        row["native_flatten_scopes"] = sum(names[n] for n in ("aten::flatten_dense_tensors", "aten::_flatten_dense_tensors"))
        row["native_fast_path_scopes"] = names[MARK]
        row["cat_scopes"] = names["aten::cat"]
        with output.open("a") as stream:
            print(json.dumps(row), file=stream)
        return result
    optimizer.flatten = observe
    return state
