"""Route eligible mixed NPU/Meta calls made through the stock PyTorch helper."""
from functools import wraps
import os
from pathlib import Path

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode

_calls=0
_accepted=0


def counters():
    return dict(calls=_calls,eligible=_accepted,fallback=_calls-_accepted)


def install():
    if os.getenv("NPU_DYNAMIC_META_UNFLATTEN") != "1":
        return
    torch.ops.load_library(str(Path(__file__).with_name("dynamic_meta_unflatten.so")))
    original = torch._utils._unflatten_dense_tensors
    native = torch.ops.npu_dynamic_meta_views.try_unflatten.default

    @wraps(original)
    def unflatten(flat, tensors):
        global _calls, _accepted
        _calls+=1
        if (type(flat) is not torch.Tensor or type(tensors) not in (tuple, list) or len(tensors) < 32 or
                torch._C._is_torch_function_mode_enabled() or _get_current_dispatch_mode() is not None or
                torch._C._are_functorch_transforms_active() or torch.compiler.is_compiling() or
                torch.jit.is_tracing() or torch.is_inference_mode_enabled() or
                flat.device.type != "npu" or flat.layout != torch.strided or flat.is_nested or
                flat.dtype != torch.bfloat16 or flat.dim() != 1 or flat.stride(0) != 1 or
                type(tensors[0]) is not torch.Tensor or tensors[0].device.type != "meta" or
                any(type(t) is not torch.Tensor for t in tensors)):
            return original(flat,tensors)
        # Respect an explicitly installed Meta implementation, including one
        # registered after torch_npu was imported.
        op = "aten::unflatten_dense_tensors"
        if (torch._C._dispatch_has_kernel_for_dispatch_key(op,"AutogradMeta") or
                torch._C._dispatch_has_kernel_for_dispatch_key(op,"Meta")):
            return original(flat,tensors)
        import torch_npu
        if torch_npu.get_npu_format(flat) != 2:
            return original(flat,tensors)
        outputs = native(flat,tensors)
        if outputs:
            _accepted+=1
            return tuple(outputs)
        return original(flat,tensors)

    torch._utils._unflatten_dense_tensors = unflatten
