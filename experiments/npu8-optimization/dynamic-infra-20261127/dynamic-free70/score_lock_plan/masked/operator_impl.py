"""Eager causal-score pointwise kernels between two unchanged native matmuls."""
from collections import Counter
import hashlib
import os
import re
import threading

import torch
import torch.nn.functional as F
import torch_npu
from triton.backends.ascend import driver
import fold_probe
from kernels import causal_score_forward, causal_score_backward

COUNTS = Counter()
PLANS = {}
EVIDENCE = {}
TABLES = {}
LOCK = threading.Lock()


def tables(device):
    with LOCK:
        if device not in TABLES:
            codes = torch.arange(65536, dtype=torch.int32).short().view(torch.bfloat16).to(device)
            x = codes.float()
            sig = torch.ones_like(x) / ((-x).exp() + 1)
            slopes = sig * (1 + x * (1 - sig))
            creator = torch.npu.current_stream(device)
            TABLES[device] = (F.silu(codes), slopes, creator, creator.record_event())
            COUNTS["table_builds"] += 1
        forward, backward, stream, event = TABLES[device]
        current = torch.npu.current_stream(device)
        if current != stream:
            current.wait_event(event)
        return forward, backward


def prepare(binary, kernel, dtypes):
    names = list(kernel.arg_names)
    idx = lambda k: names.index(k) if isinstance(k, str) else k
    signature = {idx(k): str(v) for k, v in binary.src.signature.items()}
    meta = fold_probe.values(binary.metadata)
    name = kernel.fn.__name__
    if (signature != dict(enumerate(dtypes)) or binary.src.constants or meta["name"] != name
            or meta["parallel_mode"] != "simd" or meta["force_simt_only"]
            or int(meta.get("coalesce_factor") or 1) != 1):
        raise ValueError(f"unexpected score kernel signature or metadata {name}: {signature}, {meta}")
    witness = fold_probe.witness_eager_launcher(binary)
    driver_text = driver.generate_npu_wrapper_src({}, signature, binary.metadata)
    arg_text = re.search(r"}\s*args\s*=\s*{(.*?)};", driver_text, re.S).group(1)
    caps = re.findall(r"blockNum\s*=\s*std::min\(blockNum,\s*\(uint32_t\)(\d+)\)", driver_text)
    blocks = re.findall(r"uint32_t blockNum\s*=\s*gridX\s*\*\s*gridY\s*\*\s*gridZ", driver_text)
    if (witness["driver_py_sha256"] != "b39118053062f3fc6d7f1047f8c9eb229de47a2b6106d8119cc7087f5b38818e"
            or not witness["uses_ffts"] or not witness["actual_null_resource_slots"]
            or witness["allocates_workspace"] or witness["allocates_sync_lock"]
            or len(caps) != len(blocks) or set(caps) != {"40"}
            or re.findall(r"\(grid([XYZ])\)", arg_text) != list("XYZ")
            or re.findall(r"static_cast<void\*>\(arg(\d+)\)", arg_text) != list(map(str, range(len(names))))):
        raise ValueError(f"unexpected score kernel argument order: {witness} {arg_text}")
    elf, = [v for v in binary.asm.values() if isinstance(v, bytes) and v.startswith(b"\x7fELF")]
    EVIDENCE[name] = dict(witness=witness, initializer=arg_text, binary_sha256=hashlib.sha256(elf).hexdigest())
    return torch_npu._C._npu_inductor_make_fast_launch_plan(
        name, binary.function, ("tensor",) * len(names), False, int(meta.get("shared_mem_dynamic_size") or 0), False, True)


def launch(kernel, tensors, native):
    x = tensors[0]
    batch, _, length, _ = x.shape
    shape = (length, batch, 1)
    device = x.device
    key = (kernel.fn.__name__, device)
    saved = PLANS.get(key)
    if (not native or not saved or torch.autograd.profiler._is_profiler_enabled
            or fold_probe.active_hooks(saved[0]) or any(t.data_ptr() % 16 for t in tensors)):
        binary = kernel[shape](*tensors)
        COUNTS["jit"] += 1
        if native and not saved and all(t.data_ptr() % 16 == 0 for t in tensors):
            types = tuple("*fp32" if t.dtype == torch.float32 else "*bf16" for t in tensors)
            PLANS[key] = (binary, prepare(binary, kernel, types))
            COUNTS["prepared"] += 1
        return
    if torch.npu.current_device() != device.index or any(t.device != device for t in tensors):
        raise ValueError("mixed device")
    stream = torch_npu._C._npu_getCurrentRawStreamNoWait(device.index)
    if batch * length > 40:
        torch_npu._C._npu_hstu_fast_launch_with_physical_cap(saved[1], stream, *shape, tensors, 40)
        COUNTS["capped"] += 1
    else:
        torch_npu._C._npu_inductor_fast_launch_with_plan(saved[1], stream, *shape, tensors)
        COUNTS["direct"] += 1


def stock(raw):
    length = raw.shape[-1]
    valid = torch.ones(length, length, dtype=torch.bool, device=raw.device).tril()
    return F.silu(raw * .25).masked_fill(~valid, 0.0) / float(length)


class Scores(torch.autograd.Function):
    @staticmethod
    def forward(ctx, raw, native):
        ctx.native = native
        ctx.save_for_backward(raw)
        forward, _ = tables(raw.device)
        out = torch.empty_like(raw)
        launch(causal_score_forward, (raw, forward, out), native)
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        raw, = ctx.saved_tensors
        if not grad.is_contiguous():
            COUNTS["grad_contiguous"] += 1
            grad = grad.contiguous()
        _, slopes = tables(raw.device)
        out = torch.empty_like(raw)
        launch(causal_score_backward, (raw, grad, slopes, out), ctx.native)
        return out, None


def apply(raw, native):
    b, h, m, n = raw.shape
    if (raw.device.type != "npu" or raw.dtype != torch.bfloat16 or h != 4 or m != n
            or not (1 <= b <= 4 and 12 <= m <= 32) or not raw.is_contiguous()):
        COUNTS["layout_fallback"] += 1
        return stock(raw)
    return Scores.apply(raw, native)
