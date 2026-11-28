"""Compare initialized values, geometry and layout errors for two native factory paths."""
import json
import os
from pathlib import Path
import sys

out, value = sys.argv[1:]
os.environ["TORCH_NPU_EMPTY_SINGLE_RESTRIDE"] = value
os.environ["TORCH_NPU_EMPTY_STRIDE_AUDIT"] = "1"
import torch
import torch_npu

rows = []
for shape in ((), (0,), (1,), (3,0,4), (1,19,4), (2,14,4), (4,32,4), (1,2,1,3), (2,)*7):
    for dtype in (torch.float32, torch.bfloat16, torch.int64):
        expected = torch.full(shape, 3, dtype=dtype)
        for kind in ("empty", "empty_with_format"):
            factory = torch.empty if kind == "empty" else torch_npu.empty_with_format
            a = factory(shape, dtype=dtype, device="npu")
            b = factory(shape, dtype=dtype, device="npu")
            a.fill_(3)
            b.fill_(7)
            assert a.is_contiguous() and a.stride() == expected.stride() and a.storage_offset() == 0
            assert torch.equal(a.cpu(), expected)
            rows.append([kind, str(dtype), shape, a.stride(), a.numel(), torch_npu.get_npu_format(a)])
errors = []
for args, kwargs in ((((-1,),),{}), (((1,2,3,4),), {"memory_format":torch.channels_last})):
    try:
        torch.empty(*args, device="npu", **kwargs)
    except RuntimeError as exc:
        errors.append(str(exc).splitlines()[0])
    else:
        raise AssertionError((args, kwargs))
Path(out).write_text(json.dumps(dict(rows=rows,errors=errors),indent=2))
