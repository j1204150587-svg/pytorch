"""Count synchronous Host submission work inside the two actual dynamic input ranges."""
import argparse
from collections import Counter
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path


def extract(leg):
    trace = leg / "trace_view.json.gz"
    with gzip.open(trace, "rt", encoding="utf-8") as stream:
        data = json.load(stream)
    events = data if isinstance(data, list) else data["traceEvents"]
    intervals = []
    for event in events:
        if event.get("ph") == "X":
            start = Decimal(str(event["ts"]))
            duration = Decimal(str(event["dur"]))
            intervals.append((start, start + duration, event))
    steps = [(start, end, event["name"]) for start, end, event in intervals
             if event["name"].startswith("ProfilerStep#")]
    rows = []
    for start, end, event in intervals:
        if event["name"] != "input_generation_and_h2d":
            continue
        step, = [name for lo, hi, name in steps if lo <= start and end <= hi]
        calls = [(e["name"], float(hi-lo)) for lo, hi, e in intervals if start <= lo and hi <= end]
        counts = Counter(name for name, duration in calls)
        names = ("AscendCL@aclrtMemcpy", "AscendCL@aclrtSynchronizeStream", "aten::_to_copy",
                 "aten::split_with_sizes", "aten::view")
        rows.append({"step": step, "input_scope_us": float(end-start), "counts": {n: counts[n] for n in names},
                     "acl_copy_us": [dt for n, dt in calls if n == "AscendCL@aclrtMemcpy"],
                     "acl_synchronize_us": [dt for n, dt in calls if n == "AscendCL@aclrtSynchronizeStream"]})
    if len(rows) != 2:
        raise ValueError(f"expected two profiled input ranges, got {rows}")
    return {"sha256": hashlib.sha256(trace.read_bytes()).hexdigest(), "steps": rows}


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pair", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    result = {leg: extract(args.pair / leg) for leg in ("01-A", "02-B")}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
