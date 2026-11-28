"""Compare observed Host scope overlap to the Free budget at a fixed active duration."""
import argparse
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path

SCOPES = ("input_generation_and_h2d", "hstu_forward_and_loss", "deepspeed_backward",
          "optimizer_step_including_native_deepspeed_norm")


def span(event):
    start = int(Decimal(str(event["ts"])) * 1000)
    return start, start + int(Decimal(str(event["dur"])) * 1000)


def analyze(leg):
    trace = leg / "trace_view.json.gz"
    with gzip.open(trace, "rt", encoding="utf8") as f:
        data = json.load(f)
    events = data if isinstance(data, list) else data["traceEvents"]
    intervals = sorted(span(e) for e in events if e.get("ph") == "X" and e["name"] == "Free")
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    steps = {int(e["name"].split("#")[-1]): span(e) for e in events
             if e.get("ph") == "X" and e["name"].startswith("ProfilerStep#")}
    raw = list(csv.DictReader((leg / "step_trace_time.csv").open(encoding="utf-8-sig")))
    if set(steps) != {4, 5, 6} or [int(r["Step"]) for r in raw] != [4, 5, 6]:
        raise ValueError(f"unexpected Level0 window in {leg}")
    rows = []
    for row in raw:
        step = int(row["Step"])
        left, right = steps[step]
        overlap = {}
        regions = []
        for name in SCOPES:
            (a, b), = [span(e) for e in events if e.get("ph") == "X" and e["name"] == name
                       and left <= span(e)[0] < right]
            regions.append((a, b))
            overlap[name] = sum(max(0, min(b, hi) - max(a, lo)) for lo, hi in merged) / 1e6
        if any(b > c for (_, b), (c, _) in zip(sorted(regions), sorted(regions)[1:])):
            raise ValueError("overlapping training phases")
        free, stage, computing = (float(row[k]) / 1000 for k in ("Free", "Stage", "Computing"))
        rows.append(dict(step=step, free_ms=free, stage_ms=stage, computing_ms=computing,
                         free_in_scopes_ms=overlap, official_free_minus_scopes_ms=free-sum(overlap.values())))
    free = sum(x["free_ms"] for x in rows) / 3
    stage = sum(x["stage_ms"] for x in rows) / 3
    limit = (stage - free) * 7 / 3
    return dict(trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest(), steps=rows,
                avg_free_ms=free, avg_stage_ms=stage, share=free/stage,
                free_ms_at_exactly_70pct_if_nonfree_ms_fixed=limit,
                reduction_ms_to_exactly_70pct_if_nonfree_ms_fixed=free-limit,
                mean_exposed_ms={s:sum(r["free_in_scopes_ms"][s] for r in rows)/3 for s in SCOPES})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("legs", type=Path, nargs="+")
    args = p.parse_args()
    args.output.write_text(json.dumps({str(l.parent.name + "/" + l.name): analyze(l) for l in args.legs}, indent=2))
