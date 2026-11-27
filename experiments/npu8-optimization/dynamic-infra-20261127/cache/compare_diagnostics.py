"""Recompute paired Mode 1 cache counts by step, operator and schedule subset."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


FIELDS = ("hit", "miss", "ineligible", "miss_zero_key", "first_hit", "first_miss",
          "within_step_hit", "within_step_miss", "earlier_step_hit", "earlier_step_miss")


def load(directory):
    def read(name):
        with (directory / name).open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))
    steps = read("steps.csv")
    operators = read("operators.csv")
    if [int(r["step"]) for r in steps] != list(range(60)):
        raise ValueError(f"{directory}: expected 60 steps")
    check = Counter()
    for row in operators:
        for key in FIELDS:
            check[int(row["step"]), key] += int(row[key])
    for step, row in enumerate(steps):
        if any(check[step, key] != int(row[key]) for key in FIELDS):
            raise ValueError(f"{directory}: per-operator counts disagree with step {step}")
    return steps, operators


def counts(rows, steps):
    result = Counter()
    for row in rows:
        if int(row["step"]) in steps:
            result.update({k: int(row[k]) for k in FIELDS})
    return result


def paired(a, b):
    calls_a, calls_b = (v["hit"] + v["miss"] for v in (a, b))
    if calls_a != calls_b:
        raise ValueError(f"lookup counts differ: {calls_a}, {calls_b}")
    return {"lookups": calls_a, "a_hit": a["hit"], "b_hit": b["hit"],
            "a_miss": a["miss"], "b_miss": b["miss"],
            "extra_hits": b["hit"] - a["hit"],
            "extra_within_step_hits": b["within_step_hit"] - a["within_step_hit"],
            "extra_earlier_step_hits": b["earlier_step_hit"] - a["earlier_step_hit"]}


def main(root, output):
    (left, a_ops), (right, b_ops) = [load(root / name) for name in ("01-A", "02-B")]
    shapes = [(int(row["batch"]), int(row["seq"])) for row in left]
    if shapes != [(int(row["batch"]), int(row["seq"])) for row in right]:
        raise ValueError("shape schedule differs")
    output.mkdir(parents=True, exist_ok=True)
    windows = {"level0_steps": range(4, 7), "steady_steps": range(10, 60)}
    windows.update({f"batch_{b}": [s for s in range(10, 60) if shapes[s][0] == b]
                    for b in sorted({b for b, _ in shapes})})
    summary = {}
    for label, steps in windows.items():
        summary[label] = dict(steps=len(steps), **paired(counts(left, steps), counts(right, steps)))
    op_names = sorted({r["op"] for r in a_ops + b_ops if int(r["step"]) >= 10})
    op_rows = []
    for op in op_names:
        a = counts((r for r in a_ops if r["op"] == op), windows["steady_steps"])
        b = counts((r for r in b_ops if r["op"] == op), windows["steady_steps"])
        op_rows.append(dict(op=op, **paired(a, b)))
    op_rows.sort(key=lambda row: -row["extra_hits"])
    with (output / "operator-deltas.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=op_rows[0])
        writer.writeheader()
        writer.writerows(op_rows)
    for name, rows in (("a", left), ("b", right)):
        summary[name + "_rss_kib"] = {"step_0_end": int(rows[0]["rss_end_kib"]),
                                      "step_59_end": int(rows[-1]["rss_end_kib"]),
                                      "peak_end": max(int(r["peak_rss_end_kib"]) for r in rows)}
    summary["unique_shapes"] = len(set(shapes))
    (output / "cache-comparison.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    main(args.root, args.output)
