"""Gate the NPU42 guarded D2D path in the training container, one process per variant."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch_npu


FAST = "NPU42GuardedD2D"
MATCH = "NPU42MatchedD2D"


def bytes_equal(tensor, expected):
    actual = tensor.detach().cpu().contiguous()
    expected = expected.contiguous()
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


def nd(tensor):
    result = tensor.to("npu")
    if torch_npu.get_npu_format(result) != 2:
        result = torch_npu.npu_format_cast(result, 2)
    assert torch_npu.get_npu_format(result) == 2
    return result


def run(enabled):
    results = {}

    def measured(name, operation, eligible):
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
            operation()
        names = [e.name for e in prof.events()]
        counts = {mark: names.count(mark) for mark in (FAST, MATCH)}
        assert counts[FAST] == (eligible if enabled else 0), (name, counts)
        assert counts[MATCH] == (0 if enabled else eligible), (name, counts)
        results[name] = counts

    pattern = torch.tensor([0, -32768, 1, 0x3F80, 0x7F80, -128, 0x7FC1, -1], dtype=torch.int16)
    bits = pattern.repeat(64).view(torch.bfloat16)
    src = nd(bits)
    dst = nd(torch.zeros_like(bits))
    measured("bf16_bits", lambda: dst.copy_(src), 1)
    bytes_equal(dst, bits)

    dst.fill_(7)
    measured("aligned_slices", lambda: dst[64:192].copy_(src[32:160]), 1)
    expected = torch.full_like(bits, 7)
    expected[64:192].copy_(bits[32:160])
    bytes_equal(dst, expected)

    fp = torch.arange(256, dtype=torch.float32)
    x = nd(fp)
    y = nd(torch.zeros_like(fp))
    measured("fp32", lambda: y.copy_(x, non_blocking=True), 1)
    bytes_equal(y, fp)

    measured("misaligned_offset", lambda: y[1:129].copy_(x[1:129]), 0)
    measured("short_tail", lambda: y[:17].copy_(x[:17]), 0)
    measured("empty", lambda: y[:0].copy_(x[:0]), 0)
    measured("same_tensor", lambda: y.copy_(y), 0)
    measured("noncontiguous", lambda: y[::2].copy_(x[::2]), 0)
    bytes_equal(y, fp)
    measured("same_storage_disjoint", lambda: y[:32].copy_(y[64:96]), 0)
    expected = fp.clone()
    expected[:32] = fp[64:96]
    bytes_equal(y, expected)

    def overlap():
        try:
            y[:128].copy_(y[64:192])
        except RuntimeError:
            return
        raise AssertionError("partial overlap was accepted")

    measured("partial_overlap_error", overlap, 0)
    measured("broadcast", lambda: y.view(2, 128).copy_(x[:128]), 0)
    bytes_equal(y.view(2, 128), fp[:128].repeat(2, 1))
    bf = nd(fp.to(torch.bfloat16))
    measured("dtype_cast", lambda: y.copy_(bf), 0)
    bytes_equal(y, fp)

    torch.npu.synchronize()
    stream = torch.npu.Stream()
    with torch.npu.stream(stream):
        measured("foreign_allocation_stream", lambda: y.copy_(x), 0)
    stream.synchronize()
    bytes_equal(y, fp)

    # Force rapid reuse of short-lived same-stream sources and targets before a final fence.
    snapshots = []
    template = nd(torch.zeros(1024, dtype=torch.bfloat16))
    for index in range(16):
        a = torch.empty_like(template).fill_(index + 1)
        b = torch.empty_like(a)
        b.copy_(a)
        del a
        torch.empty_like(b).fill_(-3)
        snapshots.append(b + 1)
        del b
        torch.empty_like(template).fill_(-9)
    for index, output in enumerate(snapshots):
        bytes_equal(output, torch.full((1024,), index + 2, dtype=torch.bfloat16))
    results["immediate_release_iterations"] = len(snapshots)

    # Allocate before capture and on the capture stream itself. A zero hit must
    # be due to the capture guard, not a mismatched allocation stream.
    capture_stream = torch.npu.Stream()
    with torch.npu.stream(capture_stream):
        cap_src = nd(fp)
        cap_dst = nd(torch.zeros_like(fp))
        measured("capture_stream_before_capture", lambda: cap_dst.copy_(cap_src), 1)
        capture_stream.synchronize()
        graph = torch_npu.npu.NPUGraph()

        def capture():
            with torch_npu.npu.graph(graph, stream=capture_stream):
                cap_dst.copy_(cap_src)

        measured("graph_capture", capture, 0)
        cap_src.fill_(5)
        graph.replay()
        capture_stream.synchronize()
        bytes_equal(cap_dst, torch.full_like(fp, 5))
        graph.reset()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("A", "B"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    enabled = args.variant == "B"
    assert os.environ["TASK_QUEUE_ENABLE"] == "1"
    assert os.environ["NPU_EXPERIMENT_GUARDED_D2D"] == str(int(enabled))
    assert os.environ["NPU_EXPERIMENT_D2D_MATCH_MARK"] == "1"
    torch.npu.set_device(0)
    result = {"variant": args.variant, "torch_npu": torch_npu.__file__, "cases": run(enabled)}
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
