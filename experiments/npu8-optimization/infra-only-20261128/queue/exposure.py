"""Identify post-service intervals with no visible enqueue on any producer thread."""
from bisect import bisect_left, bisect_right
from collections import Counter
from decimal import Decimal
import argparse
import gzip
import json
from pathlib import Path


def ns(value):
    return int(Decimal(str(value)) * 1_000)


def measure(path):
    with gzip.open(path, "rt", encoding="utf-8-sig") as stream:
        trace = json.load(stream)
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    queues = {"enqueue": [], "dequeue": []}
    markers = []
    raw_free = []
    enqueue_end = {}
    acl_launch = []
    for row in events:
        if row.get("ph") != "X":
            continue
        cat = row.get("cat")
        name = row["name"]
        if cat not in queues and not name.startswith(("ProfilerStep#", "AscendCL@aclrtLaunchKernel")) and name != "Free":
            continue
        t = ns(row["ts"])
        span = (t, t + ns(row["dur"]), row["tid"], name, row.get("args", {}).get("correlation_id"))
        if name == "Free":
            raw_free.append(span[:2])
        elif cat in queues:
            queues[cat].append(span)
            if cat == "enqueue":
                enqueue_end[span[4]] = span[1]
        elif name.startswith("AscendCL@aclrtLaunchKernel"):
            acl_launch.append(span)
        else:
            markers.append(span)
    starts = sorted(r[0] for r in queues["enqueue"])
    ends = sorted(r[1] for r in queues["enqueue"])
    free = []
    for a, b in sorted(raw_free):
        if free and a <= free[-1][1]:
            free[-1] = (free[-1][0], max(free[-1][1], b))
        else:
            free.append((a, b))
    free_ends = [r[1] for r in free]
    def free_overlap(a, b):
        i = bisect_right(free_ends, a)
        total = 0
        while i < len(free) and free[i][0] < b:
            total += min(b, free[i][1]) - max(a, free[i][0])
            i += 1
        return total
    reply = []
    for left, right, _, step, _ in sorted(markers):
        rows = sorted(r for r in queues["dequeue"] if left <= r[0] < right)
        if not rows:
            continue
        if len(set(r[2] for r in rows)) != 1:
            raise ValueError("requires single shared repository")
        idle = []
        for last, following in zip(rows, rows[1:]):
            stop, start = last[1], following[0]
            if stop >= start:
                continue
            overlaps = bisect_left(starts, start) - bisect_right(ends, stop)
            if not overlaps:
                next_ready = following[4] in enqueue_end and enqueue_end[following[4]] <= stop
                idle.append((last[3], start - stop, free_overlap(stop, start), next_ready))
        acl = sorted((b-a)/1000 for a,b,_,_,_ in acl_launch if left <= a < right)
        ready_idle = [r for r in idle if r[3]]
        reply.append(dict(step=step, dequeue=len(rows), adjacent_without_any_enqueue=len(idle),
                          unmarked_interval_ms=sum(r[1] for r in idle) / 1e6,
                          official_free_overlap_ms=sum(r[2] for r in idle) / 1e6,
                          already_queued_intervals=len(ready_idle),
                          already_queued_unmarked_ms=sum(r[1] for r in ready_idle) / 1e6,
                          already_queued_free_overlap_ms=sum(r[2] for r in ready_idle) / 1e6,
                          single_acl_launch_us={k:acl[int((len(acl)-1)*q)] for k,q in
                                                (("p50",.50),("p95",.95),("p99",.99),("max",1))} if acl else {},
                          by_preceding_request=dict(Counter(r[0] for r in idle).most_common(10))))
    return reply


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.write_text(json.dumps(measure(args.trace), indent=2) + "\n")
