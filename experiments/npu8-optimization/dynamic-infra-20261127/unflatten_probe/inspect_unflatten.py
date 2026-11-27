"""Observe actual native #43 guards, registrations and dispatch in an isolated run."""
from collections import Counter
import hashlib
import json
import mmap
import os
from pathlib import Path
import sys
import torch
import torch_npu

OP = "aten::unflatten_dense_tensors"
MARK = "NPUUnflattenContiguousViews"


def describe(t):
    base = t._base
    return dict(device=str(t.device), dtype=str(t.dtype), layout=str(t.layout), type=type(t).__name__,
                strides=list(t.stride()), shape=list(t.shape), storage_offset=t.storage_offset(),
                keys=str(torch._C._dispatch_keys(t)), inference=t.is_inference(),
                requires_grad=t.requires_grad, base_requires_grad=None if base is None else base.requires_grad,
                forward_tangent=torch.autograd.forward_ad.unpack_dual(t).tangent is not None,
                named=t.has_names(), nested=t.is_nested, is_neg=t.is_neg(), is_conj=t.is_conj(),
                npu_format=int(torch_npu.get_npu_format(t)) if t.device.type == "npu" else None)


def registration():
    return torch._C._dispatch_has_kernel_for_dispatch_key(OP, "AutogradPrivateUse1")


def dispatch_check(original, flat, templates):
    """A separate diagnostic call replaces an already-registered kernel, then restores it."""
    if not registration():
        return {"installed": False, "reason": "native kernel absent"}
    before = torch._C._dispatch_dump_table(OP)
    count = 0

    def sentinel(flat, tensors):
        nonlocal count
        count += 1
        return []

    probe = torch.library.Library("aten", "IMPL", "AutogradPrivateUse1")
    result = {}
    try:
        probe.impl("unflatten_dense_tensors", sentinel)
        keys = torch._C.DispatchKeySet(torch._C.DispatchKey.AutogradPrivateUse1)
        torch.ops.aten.unflatten_dense_tensors.default.redispatch(keys, flat, templates)
        result["forced_npu_calls"] = count
        before_call = count
        original(flat, templates)
        result["normal_npu_calls"] = count - before_call
    except Exception as error:
        result["probe_error"] = repr(error)
    finally:
        probe._destroy()
    result["restored"] = torch._C._dispatch_dump_table(OP) == before
    if not result["restored"]:
        raise RuntimeError("failed to restore native operator registration")
    return result


def install(optimizer, rank_dir):
    real_unflatten = optimizer.unflatten
    library, = sorted({Path(row.split()[-1]) for row in Path("/proc/self/maps").read_text().splitlines()
                      if row.split()[-1].endswith("/libtorch_npu.so")})
    with library.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as image:
        marker_present = image.find(MARK.encode()) >= 0
        digest = hashlib.sha256(image).hexdigest()
    output = Path(rank_dir) / "unflatten-probe.jsonl"
    with output.open("w") as stream:
        print(json.dumps(dict(library=str(library), sha256=digest, marker_in_library=marker_present,
                              native_npu_registration=registration(),
                              switch=os.environ.get("NPU_EXPERIMENT_UNFLATTEN_VIEWS"),
                              callable_name=real_unflatten.__qualname__,
                              dispatch_table=torch._C._dispatch_dump_table(OP))), file=stream)
    state = {"step": -1, "shape": None}

    def observe(flat, templates):
        flat_info = describe(flat)
        template_meta = Counter()
        total = 0
        empty = 0
        for t in templates:
            sample = describe(t)
            for k in ("strides", "shape", "storage_offset"):
                del sample[k]
            template_meta[json.dumps(sample, sort_keys=True)] += 1
            total += t.numel()
            empty += t.numel() == 0
        caller = sys._getframe(1).f_code.co_name
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            result = real_unflatten(flat, templates)
        names = Counter(event.name for event in prof.events())
        row = dict(step=state["step"], shape=state["shape"], caller=caller, flat=flat_info,
                   templates=len(templates), template_classes=[dict(count=n, metadata=json.loads(k)) for k,n in template_meta.items()],
                   total_template_elements=total, empty_templates=empty, padding=flat.numel()-total,
                   grad_enabled=torch.is_grad_enabled(), inference_mode=torch.is_inference_mode_enabled(),
                   tls_include=str(torch._C._dispatch_tls_local_include_set()), tls_exclude=str(torch._C._dispatch_tls_local_exclude_set()),
                   first_template=describe(templates[0]) if templates else None,
                   events=dict(native_marker=names[MARK], native_op=names[OP],
                               narrow=names["aten::narrow"], slice=names["aten::slice"],
                               as_strided=names["aten::as_strided"], view=names["aten::view"]))
        if state["step"] == 7 and caller == "_update_model_bit16_weights":
            row["actual_dispatch_probe"] = dispatch_check(real_unflatten, flat, templates)
        with output.open("a") as stream:
            print(json.dumps(row), file=stream)
        return result

    optimizer.unflatten = observe
    return state
