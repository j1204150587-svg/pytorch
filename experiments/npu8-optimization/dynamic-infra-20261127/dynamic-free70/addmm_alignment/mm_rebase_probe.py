"""Check the compact descriptor for the eight read-only HSTU projection weights in backward."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from probe_128 import pair
import torch
import torch_npu


def run(args, report):
    torch.npu.set_device(0)
    layouts = json.loads(args.geometry.read_bytes())["per_call"]
    schedule = json.loads(args.schedule.read_bytes())
    steps = [[r["original"] for r in layouts if r["step"] == i and r["has_projection_parameter"]]
             for i in range(len(schedule))]
    assert len(steps) == 60 and all(len(rows) == 8 for rows in steps)
    g = torch.Generator().manual_seed(1234)
    bank_size = steps[0][0][1][4]
    flats = [torch.randn(bank_size, generator=g).bfloat16().npu() for _ in range(2)]
    weights = []
    inputs = []
    for i, rows in enumerate(steps):
        layer_weights = []
        layer_inputs = []
        for x, w, y in rows:
            assert x[3] == 0 and x[1][1] == w[1][0] and y[1] == [x[1][0], w[1][1]]
            assert x[2] == [x[1][1], 1] and w[2] == [w[1][1], 1]
            layer_inputs.append(torch.randn(x[1], generator=g).bfloat16().npu())
            layer_weights.append(pair(flats[i % 2], dict(shape=w[1], stride=w[2], storage_element_offset=w[3])))
        inputs.append(layer_inputs)
        weights.append(layer_weights)
    torch.npu.synchronize()

    def backward_projections(i, leg):
        return [torch.mm(x, w[int(leg == "B")]) for x, w in zip(inputs[i], weights[i], strict=True)]

    if args.leg:
        for i in range(10):
            backward_projections(i, args.leg)
        torch.npu.synchronize()
        start = time.perf_counter_ns()
        for i in range(10, 60):
            backward_projections(i, args.leg)
        torch.npu.synchronize()
        report["us_per_eight_mm"] = (time.perf_counter_ns() - start) / 50000
        return
    # Fresh process timing is separate. Here both descriptor forms use one physical stream.
    profiler = torch_npu.profiler
    profile_dir = args.output.parent / "same-stream-profile"
    with profiler.profile(activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
                          experimental_config=profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level1),
                          on_trace_ready=profiler.tensorboard_trace_handler(str(profile_dir))) as prof:
        for leg in "ABBA":
            for i in (4, 5):
                with torch.profiler.record_function(f"ReadOnlyMM_{leg}_{i}"):
                    backward_projections(i, leg)
                    torch.npu.synchronize()
        prof.step()
    report["same_stream_profile"] = str(profile_dir)
    streams = [torch.npu.current_stream(), torch.npu.Stream()]
    report["shape_checks"] = []
    for i, size in enumerate(schedule):
        with torch.npu.stream(streams[i % 2]):
            a = backward_projections(i, "A")
            b = backward_projections(i, "B")
            diffs = [int((x.cpu().view(torch.int16) != y.cpu().view(torch.int16)).sum())
                     for x, y in zip(a, b, strict=True)]
        report["shape_checks"].append(dict(shape=size, differing_bf16_words=diffs))
        if any(diffs):
            raise ValueError(f"mismatched dx for {size}")
    report["exact"] = True
    report["windows"] = []
    for i, leg in enumerate("ABBA"):
        dest = args.output.with_name(f"mm-{i}-{leg}.json")
        subprocess.run([sys.executable, "-B", __file__, "--geometry", str(args.geometry), "--schedule",
                        str(args.schedule), "--leg", leg, "--output", str(dest)], check=True)
        report["windows"].append(dict(leg=leg, **json.loads(dest.read_bytes())))



if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for field in ("geometry", "schedule", "output"):
        cli.add_argument("--" + field, required=True, type=Path)
    cli.add_argument("--leg", choices=("A", "B"))
    args = cli.parse_args()
    result = {}
    try:
        run(args, result)
    finally:
        args.output.write_text(json.dumps(result, indent=2))
