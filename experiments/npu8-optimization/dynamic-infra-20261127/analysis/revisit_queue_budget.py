"""Measure wall time covered by overlapping embedding waits and earlier consumer work."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path

from analyze_dynamic_trace import Span, is_work, write_csv


def union(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def intersect(left, right):
    """Intersect two chronologically sorted lists of disjoint intervals."""
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        if max(a, c) < min(b, d):
            out.append((max(a, c), min(b, d)))
        if b <= d:
            i += 1
        if d <= b:
            j += 1
    return out


def length(windows):
    return sum(end - start for start, end in windows)


def longest_uncovered(windows, covered):
    index = 0
    longest = 0
    for start, end in windows:
        cursor = start
        while index < len(covered) and covered[index][0] < end:
            a, b = covered[index]
            longest = max(longest, a - cursor)
            cursor = b
            index += 1
        longest = max(longest, end - cursor)
    return longest


def measure(leg, analysis):
    with gzip.open(leg / "trace_view.json.gz", "rt", encoding="utf-8-sig") as handle:
        trace = json.load(handle)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    pids = {e["args"]["name"]: e["pid"] for e in events if e["name"] == "process_name"}
    enq, deq, steps = {}, {}, {}
    hardware = []
    for event in events:
        if event["ph"] != "X":
            continue
        if event.get("cat") in {"enqueue", "dequeue"}:
            items = enq if event["cat"] == "enqueue" else deq
            key = event["args"]["correlation_id"]
            if key in items:
                raise ValueError(f"duplicate {event['cat']} correlation {key}")
            items[key] = Span.read(event)
        elif event["name"].startswith("ProfilerStep#"):
            steps[int(event["name"].split("#")[1])] = Span.read(event)
        elif event["pid"] == pids["Ascend Hardware"] and is_work(event["args"].get("Task Type", "")):
            span = Span.read(event)
            hardware.append((span.start, span.end))
    del trace, events
    busy_device = union(hardware)
    shape_rows = json.loads((leg / "rank000/metrics.json").read_bytes())["steps"]
    shapes = {r["step"]: (r["batch_size"], r["seq_len"]) for r in shape_rows}
    windows_by_step = defaultdict(list)
    with gzip.open(analysis / "device_gaps.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["state"] == "linked":
                windows_by_step[int(row["step"])].append(row)
    summary, breakdown = [], []
    for step, scope in sorted(steps.items()):
        for op in ("aclnnEmbedding", "aclnnEmbeddingDenseBackward"):
            requests = {q: (e.end, deq[q].start) for q, e in enq.items()
                        if scope.start <= e.start < scope.end and q in deq and e.name == f"Enqueue@{op}"}
            if len(requests) != 1027 or len({deq[q].tid for q in requests}) != 1:
                raise ValueError(f"unexpected request count or consumer lanes at {step}/{op}")
            lane_tid = deq[next(iter(requests))].tid
            service = defaultdict(list)
            for record in deq.values():
                if record.tid == lane_tid and scope.start < record.end and record.start < scope.end:
                    service[record.name.split("@", 1)[-1]].append((record.start, record.end))
            service = {key: union(windows) for key, windows in service.items()}
            full_service = union(pair for windows in service.values() for pair in windows)
            if length(full_service) != sum(length(windows) for windows in service.values()):
                raise ValueError("overlapping dequeue records on one lane")
            for cohort, threshold in (("all_positive", 0), ("above_1ms", 1_000_000)):
                selected = {q: w for q, w in requests.items() if w[1] - w[0] > threshold}
                waits = union(selected.values())
                marked = intersect(waits, full_service)
                device_busy = intersect(waits, busy_device)
                own_gap = 0
                for row in windows_by_step[step]:
                    if int(row["queue_id"]) not in selected:
                        continue
                    a = scope.start + int(row["step_offset_ns"])
                    b = a + int(row["gap_ns"])
                    start, end = selected[int(row["queue_id"])]
                    overlap = max(0, min(end, b) - max(start, a))
                    if overlap != int(row["visible_queue_wait_ns"]):
                        raise ValueError("gap/queue join disagrees with trace")
                    own_gap += overlap
                wait_ns = length(waits)
                if own_gap > wait_ns - length(device_busy):
                    raise ValueError("own exposed wait exceeds device idle")
                base = dict(step=step, batch=shapes[step][0], seq=shapes[step][1], op=op, cohort=cohort)
                times = dict(request_wait_sum=sum(b - a for a, b in selected.values()),
                             wait_union=wait_ns, recorded_consumer=length(marked),
                             outside_consumer=wait_ns - length(marked),
                             longest_contiguous_outside_consumer=longest_uncovered(waits, marked),
                             device_work=length(device_busy), device_idle=wait_ns - length(device_busy),
                             idle_during_recorded_consumer=length(marked) - length(intersect(marked, busy_device)),
                             own_successor_gap_wait=own_gap)
                summary.append(dict(**base, request_count=len(requests), selected_count=len(selected),
                                    **{k + "_ms": round(v / 1e6, 6) for k, v in times.items()}))
                for name, records in service.items():
                    duration = length(intersect(waits, records))
                    if duration:
                        breakdown.append(dict(**base, servicing_op=name, service_ms=round(duration / 1e6, 6)))
    return summary, breakdown


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="dynamic-infra experiment directory")
    args = parser.parse_args()
    analysis = args.root / "analysis"
    rows, parts = [], []
    for source, result in (("baseline-l1", "baseline-l1"), ("repeat-late/01-A", "repeat-late")):
        time_rows, service_rows = measure(args.root / "evidence" / source, analysis / result)
        rows.extend(time_rows)
        parts.extend(service_rows)
    write_csv(analysis / "repeat-late/queue_backlog.csv", rows)
    write_csv(analysis / "repeat-late/queue_backlog_services.csv", parts)
