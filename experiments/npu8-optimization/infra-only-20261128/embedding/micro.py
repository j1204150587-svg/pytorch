"""Probe alternating cached lookup and gradient calls, separately from HSTU training."""

import argparse
import ctypes
import json
import os
from pathlib import Path

import torch
import torch_npu


def native_counters():
    libs = {row.split()[-1] for row in Path("/proc/self/maps").read_text().splitlines()
            if row.split()[-1].endswith("/libtorch_npu.so")}
    if len(libs) != 1:
        raise RuntimeError(f"expected one libtorch_npu.so, got {libs}")
    lib = ctypes.CDLL(libs.pop())
    funcs = {}
    for direction in ("backward", "forward"):
        func = getattr(lib, f"npu_embedding_{direction}_rebase_counter")
        func.argtypes = (ctypes.c_uint,)
        func.restype = ctypes.c_uint64
        funcs[direction] = func
    return lambda: {key: [f(i) for i in range(4)] for key, f in funcs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    if os.environ.get("NPU_DYNAMIC_REBASE_COUNT") != "1":
        raise ValueError("set NPU_DYNAMIC_REBASE_COUNT=1 for this separate probe")
    torch.npu.set_device(0)
    fetch = native_counters()
    before = fetch()
    lookup = torch.ops.aten.embedding.default
    gradient = torch.ops.aten.embedding_dense_backward.default
    checks = []
    tables = 64
    for generation, (batch, length) in enumerate(((1, 19), (2, 14), (4, 32), (1, 19))):
        banks = []
        for owner in range(2):
            x = torch.arange(batch*length*tables*4, dtype=torch.long)
            ids = ((x*3 + x//5 + owner*7 + generation) % 16).view(batch, length, tables, 4)
            x = torch.arange(tables*4096 + 512, dtype=torch.int32)
            weight = ((x*19 + owner*17 + generation) % 251 - 125).to(torch.bfloat16)
            banks.append((ids, ids.npu(), weight, weight.npu()))
        for table in (0, 1, 15, 16, 31, 32, 48, 63):
            for owner in range(2):
                ic, inu, _, _ = banks[owner]
                _, _, wc, wn = banks[1-owner]
                i, ni = ic[:, :, table, :], inu[:, :, table, :]
                wc = wc.narrow(0, 257+table*4096, 4096).view(64, 64)
                wn = wn.narrow(0, 257+table*4096, 4096).view(64, 64)
                pad = 7 if owner else -1
                # Integer-valued gradients remain exactly representable in BF16 for this input.
                grad = ((torch.arange(i.numel()*64).view(*i.shape, 64) + table + owner) % 3 - 1).to(torch.bfloat16)
                g = grad.npu()
                old = fetch()
                assert torch.equal(lookup(wn, ni, pad, False, False).cpu(), lookup(wc, i, pad, False, False))
                mid = fetch()
                assert old["backward"] == mid["backward"]
                assert torch.equal(gradient(g, ni, 64, pad, False).cpu(), gradient(grad, i, 64, pad, False))
                assert mid["forward"] == fetch()["forward"]
                checks.append([batch, length, generation, table, owner, pad])
    target = fetch()
    i, ni = i[0], ni[0]
    assert torch.equal(lookup(wn, ni, -1, False, False).cpu(), lookup(wc, i, -1, False, False))
    assert torch.equal(gradient(g[0], ni, 64, -1, False).cpu(), gradient(grad[0], i, 64, -1, False))
    fallback = fetch()
    switches = {"backward": "NPU_DYNAMIC_REBASE_INDICES", "forward": "NPU_DYNAMIC_REBASE_FORWARD"}
    for direction, switch in switches.items():
        if os.environ.get(switch) == "1":
            assert target[direction][0]-before[direction][0] == len(checks)
            assert fallback[direction][1]-target[direction][1] == 1
            assert target[direction][2]-before[direction][2] == len(checks)*(2 if direction == "forward" else 1)
        else:
            assert before[direction] == fallback[direction]
    args.out.write_text(json.dumps(dict(before=before, target=target, fallback=fallback,
                                        checks=checks, exact_cpu=True), indent=2) + "\n")


if __name__ == "__main__":
    main()
