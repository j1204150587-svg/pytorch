"""Check #44 markers and D2D contents across explicitly ordered allocation streams."""
import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import torch_npu
from test_guarded_copy import bytes_equal, nd


SRC_FAST = "NPU44GuardedSourceD2D"
SRC_A = "NPU44MatchedSourceD2D"
SAME = "NPU42GuardedD2D"
MARKS = (SRC_FAST, SRC_A, SAME, "NPU42MatchedD2D")


def check(ok, message):
    if not ok:
        raise RuntimeError(message)


def marked(name, operation, wanted, records):
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        operation()
    names = [event.name for event in prof.events()]
    counts = {mark: names.count(mark) for mark in MARKS}
    check(all(n == int(mark == wanted) for mark, n in counts.items()), f"{name}: {counts}, expected {wanted}")
    records[name] = counts


def markers(variant):
    p, c = torch.npu.Stream(), torch.npu.Stream()
    values = torch.arange(128, dtype=torch.float32)
    seen = {}
    with torch.npu.stream(p):
        fp, bf, dst_on_p = nd(values), nd(values.to(torch.bfloat16)), nd(torch.zeros_like(values))
        fp.add_(1)
    with torch.npu.stream(c):
        dst, local = nd(torch.zeros_like(values)), nd(values)
        c.wait_stream(p)
        marked("other_source", lambda: dst.copy_(fp), SRC_FAST if variant == "B" else SRC_A, seen)
        c.synchronize()
        bytes_equal(dst, values + 1)
        marked("same_stream_42", lambda: dst.copy_(local), SAME, seen)
        bytes_equal(dst, values)
        marked("foreign_destination", lambda: dst_on_p.copy_(local), None, seen)
        marked("cross_dtype", lambda: dst.copy_(bf), None, seen)
        bytes_equal(dst, values)
        marked("cross_noncontiguous", lambda: dst[::2].copy_(fp[::2]), None, seen)
        expected = values.clone()
        expected[::2] += 1
        bytes_equal(dst, expected)
        marked("broadcast", lambda: dst.view(2, 64).copy_(fp[:64]), None, seen)
        expected = (values[:64] + 1).repeat(2)
        bytes_equal(dst, expected)
        marked("same_storage_disjoint", lambda: dst[:32].copy_(dst[64:96]), None, seen)
        expected[:32] = expected[64:96].clone()
        bytes_equal(dst, expected)
        def overlap():
            try:
                dst[:64].copy_(dst[32:96])
            except RuntimeError:
                return
            raise RuntimeError("partially overlapping copy succeeded")
        marked("partial_overlap_raises", overlap, None, seen)
        c.synchronize()
    bytes_equal(dst_on_p, values)
    return seen


def lifetime(variant, shape, dtype, explicit_record):
    producer, consumer = torch.npu.Stream(), torch.npu.Stream()
    expected = (torch.arange(shape[0] * shape[1]) % 23).reshape(shape).to(dtype)
    with torch.npu.stream(consumer):
        dst = nd(torch.zeros_like(expected))
        # This cube backlog keeps the last copy outstanding while the allocating stream requests blocks.
        m = torch.ones((2048, 2048), device="npu", dtype=torch.float16)
        out = torch.empty_like(m)
        torch.mm(m, m, out=out)
        consumer.synchronize()
    with torch.npu.stream(producer):
        src = nd(expected)
        src.add_(2)
        addr = src.data_ptr()
    with torch.npu.stream(consumer):
        consumer.wait_stream(producer)
        for _ in range(48):
            torch.mm(m, m, out=out)
        if explicit_record:
            src.record_stream(consumer)
        dst.copy_(src)
        copied = consumer.record_event()
    del src
    with torch.npu.stream(producer):
        pending = not copied.query()
        live = []
        reused_while_pending, reused_any = False, False
        for _ in range(16):
            other = torch.empty(shape, device="npu", dtype=dtype)
            if other.data_ptr() == addr:
                reused_any = True
                reused_while_pending = not copied.query()
            other.fill_(-13)
            live.append(other)
    copied.synchronize()
    producer.synchronize()
    observed = dst.cpu()
    equal = torch.equal(observed.contiguous().view(torch.uint8), (expected + 2).contiguous().view(torch.uint8))
    # Retain the pressure blocks while looking for the original block after retirement.
    with torch.npu.stream(producer):
        for _ in range(128):
            if reused_any:
                break
            other = torch.empty(shape, device="npu", dtype=dtype)
            reused_any = other.data_ptr() == addr
            live.append(other)
    strong = explicit_record or variant == "B"
    if strong:
        check(pending, f"{shape}: copy drained before deletion/reuse pressure")
        check(equal and not reused_while_pending and reused_any,
              f"{shape}: correct={equal}, reused while copy pending={reused_while_pending}, reused={reused_any}")
    return {"shape": list(shape), "dtype": str(dtype), "manual_record_stream": explicit_record,
            "pending_after_delete": pending, "original_address_observed_again": reused_any,
            "reused_while_copy_pending": reused_while_pending, "content_equal": equal}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("variant", choices=("A", "B"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    check(os.environ["TASK_QUEUE_ENABLE"] == "1" and
          os.environ["NPU_EXPERIMENT_GUARDED_D2D"] == os.environ["NPU_EXPERIMENT_D2D_RECORD_SRC_MARK"] == "1"
          and os.environ["NPU_EXPERIMENT_D2D_RECORD_SRC"] == str(int(args.variant == "B")), "wrong experiment flags")
    torch.npu.set_device(0)
    libs = {line.split(maxsplit=5)[-1].strip() for line in Path("/proc/self/maps").read_text().splitlines()
            if line.rstrip().endswith("/libtorch_npu.so")}
    check(len(libs) == 1, f"unexpected mapped PTA library: {libs}")
    lib = Path(libs.pop())
    with lib.open("rb") as stream:
        h = hashlib.sha256()
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    result = {"variant": args.variant, "package": torch_npu.__file__, "library": str(lib), "sha256": h.hexdigest(),
              "markers": markers(args.variant), "lifetime": []}
    for spec in (((64, 64), torch.bfloat16), ((128, 64), torch.float32)):
        for manual in (True, False):
            result["lifetime"].append(lifetime(args.variant, *spec, explicit_record=manual))
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    result["status"] = "passed"
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
