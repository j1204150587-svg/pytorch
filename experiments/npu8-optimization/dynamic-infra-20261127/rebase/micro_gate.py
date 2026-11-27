"""Compare per-table dense gradients to CPU and read native activation counters."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path

import torch
import torch_npu


def native_counters():
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if "/" in line and ("/libtorch_npu" in line or "libop_plugin" in line or "libopplugin" in line)}
    for path in sorted(paths):
        try:
            symbol = ctypes.CDLL(path).npu_dynamic_index_rebase_counter
        except (OSError, AttributeError):
            continue
        symbol.argtypes = (ctypes.c_uint,)
        symbol.restype = ctypes.c_uint64
        return {name:symbol(i) for i,name in enumerate(("eligible","fallback","key_uses","descriptor_uses"))}
    raise RuntimeError("native experiment counters missing from loaded libraries")


def make_ids(batch, length, tables):
    # Deliberately reuse IDs within and across positions, including padding 0.
    flat = torch.arange(batch * length * tables * 4, dtype=torch.long)
    return ((flat * 7 + flat // 3) % 29).view(batch,length,tables,4)


def check_one(ids_cpu, ids_npu, padded):
    shape = [*ids_cpu.shape, 8]
    grad = ((torch.arange(ids_cpu.numel() * 8).reshape(shape) % 5) - 2).to(torch.bfloat16)
    padding = 0 if padded else -1
    backward = torch.ops.aten.embedding_dense_backward.default
    expected = backward(grad,ids_cpu,32,padding,False)
    # All summands and partial sums are small integers representable in BF16.
    result = backward(grad.npu(),ids_npu,32,padding,False).cpu()
    if not torch.equal(result,expected):
        raise ValueError(f"gradient differs at shape {list(ids_cpu.shape)} / offset {ids_cpu.storage_offset()}")
    return hashlib.sha256(result.float().contiguous().numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.npu.set_device(0)
    before = native_counters()
    checks = []
    for b,length in ((1,19),(2,14),(2,22),(4,32),(1,19)):
        source = make_ids(b,length,128)
        device_source = source.npu()
        for table in (0,1,15,16,17,31,32,63,64,65,126,127):
            cpu_view = source[:,:,table,:]
            device_view = device_source[:,:,table,:]
            results = [check_one(cpu_view,device_view,padding) for padding in (False,True)]
            checks.append(dict(batch=b, seq_len=length, table=table, offset=cpu_view.storage_offset(), sha256=results))
    after_target = native_counters()
    # A rank-two index follows the stock descriptor and still matches CPU.
    two = torch.arange(12, dtype=torch.long).reshape(2,6) % 5
    guard_sha = check_one(two,two.npu(),True)
    torch.npu.synchronize()
    end = native_counters()
    mode = os.getenv("NPU_DYNAMIC_REBASE_INDICES","0")
    if mode == "1":
        if after_target["eligible"] <= before["eligible"] or after_target["key_uses"] <= before["key_uses"]:
            raise ValueError(f"rank-three indices did not use the candidate: {before} -> {after_target}")
        if end["eligible"] != after_target["eligible"] or end["fallback"] <= after_target["fallback"]:
            raise ValueError(f"rank-two fallback did not activate: {after_target} -> {end}")
    elif end != before:
        raise ValueError("candidate counters advanced on A")
    args.output.write_text(json.dumps(dict(mode=mode, before=before, target=after_target, end=end,
                                          checks=checks, fallback_sha256=guard_sha), indent=2))


if __name__ == "__main__":
    main()
