"""Rank device gaps in the dual-rebase leg and check observable stream-44 copy lookahead."""

import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
from itertools import groupby
import json
from pathlib import Path


def rows(path):
    with gzip.open(path, "rt", newline="") as src:
        return list(csv.DictReader(src))


def ns(x):
    return int(Decimal(str(x)) * 1000)


STAGES = ("before_aten_ns", "aten_before_workspace_ns", "workspace_ns", "aten_to_enqueue_remainder_ns",
          "enqueue_ns", "visible_queue_wait_ns", "consumer_before_api_ns", "acl_submission_api_ns", "after_api_ns",
          "before_enqueue_unresolved_ns")


def collect(data, raw):
    gaps = rows(data / "device_gaps.csv.gz")
    calls = rows(data / "queue_calls.csv.gz")
    work = rows(data / "work_tasks.csv.gz")
    result = {}
    with gzip.open(raw / "trace_view.json.gz", "rt", encoding="utf-8-sig") as src:
        trace = json.load(src)
    events = trace if isinstance(trace, list) else trace["traceEvents"]
    en, service = {}, defaultdict(list)
    for event in events:
        if event["ph"] != "X" or event.get("cat") not in ("enqueue", "dequeue"):
            continue
        cid = str(event["args"]["correlation_id"])
        if event["cat"] == "enqueue":
            en[cid] = ns(event["ts"]) + ns(event["dur"])
        else:
            service[event["tid"]].append((ns(event["ts"]), cid))
    for lane in service.values():
        lane.sort()
    del events, trace
    for step in ("4", "5"):
        considered = [g for g in gaps if g["step"] == step]
        linked = [g for g in considered if g["state"] == "linked"]
        q = {r["queue_id"]: r["op"] for r in calls if r["step"] == step}
        by_q = defaultdict(list)
        for r in work:
            if r["step"] == step and r["queue_id"]:
                by_q[r["queue_id"]].append(r)
        ids = {key for key, rs in by_q.items() if q.get(key) == "aclnnInplaceCopy" and len(rs) == 1 and
               rs[0]["stream"] == "44" and rs[0]["task_type"].startswith("SDMA") and rs[0]["task"] == "MEMCPY_ASYNC"}
        groups, ready, adjacent = [], [], 0
        for lane in service.values():
            for eligible, seq in groupby(lane, key=lambda r: r[1] in ids):
                if not eligible:
                    continue
                group = list(seq)
                groups.append(len(group))
                adjacent += sum(en[nxt[1]] <= prev[0] for prev, nxt in zip(group, group[1:]))
                i = 0
                while i < len(group):
                    j = i + 1
                    while j < len(group) and j-i < 64 and en[group[j][1]] <= group[i][0]:
                        j += 1
                    ready.append(j-i)
                    i = j
        if sum(groups) != len(ids):
            raise ValueError("stream-44 copy request missing or multiply linked to consumer")
        op_groups = defaultdict(list)
        for gap in linked:
            op_groups[gap["op"]].append(gap)
        table = []
        for op, selected in op_groups.items():
            intervals = {k: sum(int(g[k] or 0) for g in selected)/1e6 for k in STAGES}
            details = [item for item in calls if item["step"] == step and item["op"] == op]
            subgroup = defaultdict(list)
            for gap in selected:
                subgroup[gap["stream"], gap["task"]].append(gap)
            table.append(dict(op=op, gaps=len(selected), gap_ms=sum(int(g["gap_ns"]) for g in selected)/1e6,
                              onset=dict(Counter(g["first_visible"] for g in selected)), intersections_ms=intervals,
                              by_role=dict(Counter(g["role"] for g in selected)), calls=len(details),
                              workspace_calls=sum(bool(c["workspace_ns"]) for c in details),
                              gap_by_stream_and_task=[dict(stream=key[0], task=key[1], count=len(rs),
                                  gap_ms=sum(int(g["gap_ns"]) for g in rs)/1e6,
                                  workspace_ms=sum(int(g["workspace_ns"] or 0) for g in rs)/1e6,
                                  after_api_ms=sum(int(g["after_api_ns"] or 0) for g in rs)/1e6)
                                  for key, rs in sorted(subgroup.items())],
                              work_by_name=dict(Counter(t["task"] for qid, ts in by_q.items() if q.get(qid) == op for t in ts))))
        table.sort(key=lambda r: -r["gap_ms"])
        result[step] = dict(all_internal_gap_ms=sum(int(g["gap_ns"]) for g in considered)/1e6,
                            linked_ms=sum(int(g["gap_ns"]) for g in linked)/1e6,
                            unlinked_states=dict(Counter(g["state"] for g in considered if g["state"] != "linked")),
                            ranked=table, stream44=dict(calls=len(ids), runs=len(groups), longest=max(groups, default=0),
                            adjacent_ready=adjacent, grouped_sizes=dict(Counter(ready)),
                            optimistic_saved_submissions=sum(n-1 for n in ready)))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("raw", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(collect(args.data, args.raw), indent=2) + "\n")
