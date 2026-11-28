"""Test binary division broadcasting on the BF16 inputs of HSTU mean backward."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu


def source(batch, length, offset, gen, special=False):
    count = batch * length * 64
    words = torch.randn(count + offset, generator=gen).bfloat16()
    if special:
        patterns = [0, 0x8000, 1, 0x8001, 0x0080, 0x8080, 0x007f, 0x807f,
                    0x7f7f, 0xff7f, 0x7f80, 0xff80, 0x7fc1, 0xffc1, 0x7f81, 0xff81]
        signed = [x if x < 32768 else x - 65536 for x in patterns]
        words.view(torch.int16)[offset:offset + len(signed)] = torch.tensor(signed, dtype=torch.int16)
    return words.npu()[offset:].view(batch, length, 1, 64).expand(batch, length, 4, 64)


def scalar(x, _):
    return x / 4


def binary(x, denom):
    return torch.ops.aten.div.Tensor(x.narrow(2, 0, 1), denom)


def word_diff(a, b):
    a, b = a.detach().cpu().contiguous(), b.detach().cpu().contiguous()
    changed = a.view(torch.int16) != b.view(torch.int16)
    both_nan = a.isnan() & b.isnan()
    return {"different": int(changed.sum()), "both_nan": int((changed & both_nan).sum())}


def check_case(b, l, offset, gen, denom, special):
    x = source(b, l, offset, gen, special)
    old = x.cpu().clone()
    a, y = scalar(x, denom), binary(x, denom)
    y2 = binary(x, denom)
    before = y2.cpu().clone()
    row = {
        "shape": [b,l], "offset": x.storage_offset(), "special": special,
        "primary": word_diff(a, y),
        "aten_scalar": word_diff(torch.ops.aten.div.Scalar(x, 4), y),
        "cpu_tensor": word_diff(torch.ops.aten.div.Tensor(x, torch.tensor(4)), y),
        "layout_ok": list(a.size()) == list(y.size()) == [b,l,4,64] and
            a.dtype == y.dtype == torch.bfloat16 and a.stride() == y.stride() and y.is_contiguous(),
        "distinct_storage": len({x.untyped_storage().data_ptr(), a.untyped_storage().data_ptr(),
                                 y.untyped_storage().data_ptr(), y2.untyped_storage().data_ptr()}) == 4,
    }
    y.zero_()
    row["aliases_changed"] = word_diff(old, x)["different"] + word_diff(before, y2)["different"]
    return row


def timing(order, bank, denom, repeats):
    rows = []
    for leg in order:
        op = scalar if leg == "A" else binary
        torch.npu.synchronize()
        t0 = time.perf_counter_ns()
        for _ in range(repeats):
            for x in bank:
                y = op(x, denom)
        torch.npu.synchronize()
        rows.append((leg, (time.perf_counter_ns() - t0) / (1000 * len(bank) * repeats)))
    return dict(rows, order=order)


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run(args):
    schedule = json.loads(args.schedule.read_text())
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    gen = torch.Generator().manual_seed(83991)
    report = {"schedule_sha256": hashlib.sha256(args.schedule.read_bytes()).hexdigest(),
              "schedule": schedule, "output": str(args.output), "checks": [], "timings_us_per_call": {}}
    streams = [torch.npu.current_stream(), torch.npu.Stream()]
    for idx, stream in enumerate(streams):
        with torch.npu.stream(stream):
            denom = torch.full((1,1,4,1), 4, dtype=torch.bfloat16, device="npu")
            for i in (range(len(schedule)) if idx == 0 else (0,4,5, len(schedule)-1)):
                b, l = schedule[i]
                row = check_case(b, l, 0 if idx == 0 else 7, gen, denom, i in (0, 4, 5, len(schedule)-1))
                report["checks"].append(dict(row, stream_idx=idx))
            torch.npu.synchronize()
    save(args.output, report)
    denom = torch.full((1,1,4,1), 4, dtype=torch.bfloat16, device="npu")
    bank = [source(b,l,0,gen) for b,l in schedule]
    timing("ABBA", bank, denom, 1)
    for key, active, repeats in (("step4",[bank[4]],128), ("step5",[bank[5]],128), ("dynamic60",bank,2)):
        rows = [timing(order, active, denom, repeats) for order in ["AB","BA"] * args.pairs]
        report["timings_us_per_call"][key] = {"per_order": rows, "calls_per_leg": len(active)*repeats,
            "paired_median_b_minus_a": statistics.median(r["B"]-r["A"] for r in rows)}
    save(args.output, report)
    if not args.no_profile:
        p = torch_npu.profiler
        cfg = p._ExperimentalConfig(profiler_level=p.ProfilerLevel.Level1)
        with p.profile(activities=[p.ProfilerActivity.CPU,p.ProfilerActivity.NPU],
                       record_shapes=True, experimental_config=cfg,
                       on_trace_ready=p.tensorboard_trace_handler(str(args.output.parent/"profile"))):
            for step, order in [(4,"AB"), (5,"BA")]:
                for leg in order:
                    op = scalar if leg == "A" else binary
                    with torch.profiler.record_function(f"DivsProbe_{leg}_{step}"):
                        for _ in range(8):
                            y = op(bank[step], denom)
                        torch.npu.synchronize()
    report["complete"] = True
    save(args.output, report)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--output", "--out", type=Path, required=True)
    ap.add_argument("--pairs", type=int, default=3)
    ap.add_argument("--no-profile", action="store_true")
    run(ap.parse_args())
