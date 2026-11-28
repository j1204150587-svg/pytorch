"""Intersect leaf CPU markers with gaps where the linked next request is still on the producer."""

import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path


def ns(x):
    return int(Decimal(str(x)) * 1000)


def csv_rows(path):
    with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def leaves(spans):
    # Clip crossing timestamps by tracking active intervals; the latest-started
    # marker with the earliest end at a shared start is the most specific.
    points = defaultdict(list)
    for identity, (a, b, name) in enumerate(spans):
        if a < b:
            ref = (a, -b, identity, name)
            points[a].append((True, ref))
            points[b].append((False, ref))
    active, slices = set(), []
    previous = None
    for time, delta in sorted(points.items()):
        if active and previous < time:
            owner = max(active)[3]
            if slices and slices[-1][2] == owner and slices[-1][1] == previous:
                slices[-1] = slices[-1][0], time, owner
            else:
                slices.append((previous, time, owner))
        for entering, ref in delta:
            if entering:
                active.add(ref)
            else:
                active.remove(ref)
        previous = time
    return slices


def intersections(left, right, lane, ends, result):
    i = bisect_right(ends, left)
    while i < len(lane) and lane[i][0] < right:
        a, b, name = lane[i]
        result[name] += min(right, b) - max(left, a)
        i += 1


def analyze(trace_path, enriched):
    with gzip.open(trace_path, "rt", encoding="utf-8-sig") as stream:
        doc = json.load(stream)
    events = doc if isinstance(doc, list) else doc["traceEvents"]
    spans, steps, counts = defaultdict(list), {}, defaultdict(Counter)
    for event in events:
        if event.get("cat") != "cpu_op" or event.get("ph") != "X":
            continue
        a, length = ns(event["ts"]), ns(event["dur"])
        b, name, tid = a + length, event["name"], event["tid"]
        spans[tid].append((a, b, name))
        if name.startswith("ProfilerStep#"):
            steps[int(name.split("#")[-1])] = a, b
    calls = csv_rows(enriched / "queue_calls.csv.gz")
    tids = {r["queue_id"]:int(r["producer_tid"]) for r in calls}
    slices = {t: leaves(s) for t, s in spans.items() if t in set(tids.values())}
    ends = {t: [r[1] for r in s] for t, s in slices.items()}
    gaps = csv_rows(enriched / "device_gaps.csv.gz")
    result = {}
    for step, (left, right) in sorted(steps.items()):
        out = {}
        ops_latency = defaultdict(list)
        for tid, lane in slices.items():
            for a,b,name in spans[tid]:
                if left <= a < right:
                    counts[step][name] += 1
                    if name in {"aten::view","empty_tensor","aten::empty_strided"}:
                        ops_latency[name].append(b-a)
            intersections(left, right, lane, ends[tid], counts[step,"exclusive_ns"])
        out["exclusive_cpu_by_leaf"] = dict(counts[step,"exclusive_ns"].most_common(100))
        out["counts"] = dict(counts[step].most_common(150))
        out["latency_us"] = {}
        for name, durations in ops_latency.items():
            durations.sort()
            out["latency_us"][name] = {k:durations[int((len(durations)-1)*q)]/1000
                                      for k,q in (("p50",.50),("p95",.95),("p99",.99),("max",1))}
        totals = {}
        for label, stage_keys in [
            ("before_next_aten", ("before_aten_ns",)),
            ("before_next_enqueue", ("before_aten_ns","aten_before_workspace_ns","workspace_ns",
                                    "aten_to_enqueue_remainder_ns","before_enqueue_unresolved_ns")),
        ]:
            by_name = Counter()
            exposed = 0
            for g in gaps:
                if g["step"] != str(step) or g["state"] != "linked":
                    continue
                a = left + int(g["step_offset_ns"])
                length = sum(int(g.get(k) or 0) for k in stage_keys)
                if length:
                    exposed += length
                    t = tids[g["queue_id"]]
                    intersections(a, a+length, slices[t], ends[t], by_name)
            totals[label] = {"ns":exposed, "leaf_ns":dict(by_name.most_common(100))}
        out["linked_exposure"] = totals
        result[step] = out
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("enriched", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.trace,args.enriched),indent=2)+"\n")
