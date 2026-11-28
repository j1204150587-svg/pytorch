"""Separate native-operator check for exact-range writable copy rebinding."""

import argparse
import ctypes
import json
import math
import os
from pathlib import Path


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output", type=Path)
    cli.add_argument("--switch", choices=("0", "1"), required=True)
    args = cli.parse_args()
    # Only the stand-alone probe configures its own process.
    os.environ.update(NPU_INPLACE_COPY_REBASE=args.switch, NPU_DYNAMIC_REBASE_COUNT="1")
    import torch
    import torch_npu
    torch.npu.set_device(0)
    libs = {r.split()[-1] for r in Path("/proc/self/maps").read_text().splitlines()
            if r.split()[-1].endswith("/libtorch_npu.so")}
    if len(libs) != 1:
        raise ValueError(libs)
    lib, = libs
    f = ctypes.CDLL(lib).npu_inplace_copy_rebase_counter
    f.argtypes = (ctypes.c_uint,)
    f.restype = ctypes.c_uint64
    def counters():
        return [f(i) for i in range(4)]
    observations = []
    def observe(label, dst, src, kind):
        pre = counters()
        dst.copy_(src)
        post = counters()
        delta = [y-x for x,y in zip(pre,post)]
        if args.switch == "0":
            assert delta == [0,0,0,0], (label,delta)
        elif kind == "eligible":
            assert delta[:3] == [1,0,2], (label,delta)
        else:
            assert delta[:3] == [0,1,0], (label,delta)
        observations.append(dict(label=label,kind=kind,counters=delta))
    def values(count, seed, dtype):
        return ((torch.arange(count, dtype=torch.int32)*17 + seed*11) % 181 - 90).to(dtype)
    shapes = [(64,64), (1,4,64), (2,8,32), (512,), (64,64)]
    base_stream = torch.npu.current_stream()
    copy_stream = torch.npu.Stream()
    for dtype in (torch.bfloat16, torch.float32):
        for generation, shape in enumerate(shapes):
            count = math.prod(shape)
            for owner in range(2):
                off_src = 256*(owner+1)
                off_dst = 256*(3-owner)
                source = values(off_src + count + 256, generation*2+owner, dtype)
                expected = torch.full((off_dst+count+(256 if owner else 0),), -99, dtype=dtype)
                source_npu, destination_npu = source.npu(), expected.npu()
                src = source_npu.narrow(0,off_src,count).view(shape)
                dst = destination_npu.narrow(0,off_dst,count).view(shape)
                expected[off_dst:off_dst+count] = source[off_src:off_src+count]
                if owner:
                    with torch.npu.stream(copy_stream):
                        copy_stream.wait_stream(base_stream)
                        observe(f"{dtype}-{generation}-{owner}", dst, src, "eligible")
                        src.record_stream(copy_stream)
                        dst.record_stream(copy_stream)
                    base_stream.wait_stream(copy_stream)
                else:
                    observe(f"{dtype}-{generation}-{owner}", dst, src, "eligible")
                assert torch.equal(destination_npu.cpu(), expected)
                assert torch.equal(source_npu.cpu(), source)

    def run_fallback(name, src_cpu, dst_cpu, transform_src=lambda t:t, transform_dst=lambda t:t):
        s, d = src_cpu.npu(), dst_cpu.npu()
        expected = dst_cpu.clone()
        transform_dst(expected).copy_(transform_src(src_cpu))
        observe(name, transform_dst(d), transform_src(s), "ordinary")
        assert torch.equal(d.cpu(), expected)
    run_fallback("unaligned", values(1025,1,torch.float32), values(1025,0,torch.float32),
                 lambda t:t[1:513], lambda t:t[256:768])
    run_fallback("odd-length", values(257,1,torch.float32), values(257,0,torch.float32))
    run_fallback("dtype-convert", values(512,1,torch.float32), values(512,0,torch.bfloat16))
    run_fallback("transpose", values(1024,1,torch.float32), values(1024,0,torch.float32),
                 lambda t:t.view(32,32).t(), lambda t:t.view(32,32))
    bank = values(1536, 0, torch.float32)
    device = bank.npu()
    bank[768:1280].copy_(bank[0:512])
    observe("same-storage-disjoint", device[768:1280], device[:512], "ordinary")
    assert torch.equal(device.cpu(),bank)
    # Native overlap checks must still precede ACLNN for both switches.
    pre = counters()
    try:
        device[1:513].copy_(device[:512])
    except RuntimeError:
        assert counters() == pre and torch.equal(device.cpu(),bank)
    else:
        raise AssertionError("partially overlapping write unexpectedly succeeded")
    args.output.write_text(json.dumps(dict(switch=args.switch,library=lib,checks=observations,
                                           overlap_error_preserved=True,counters_final=counters()),indent=2)+"\n")


if __name__ == "__main__":
    main()
