"""Connect the largest dynamic device gaps to concurrent host activity."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import gzip
import json
from pathlib import Path

from analyze_dynamic_trace import Span, is_work, stats, write_csv


UNFLATTEN_OPS = {
    "aten::unflatten_dense_tensors", "aten::_unflatten_dense_tensors",
    "npu_dynamic_meta_views::try_unflatten",
}
UNFLATTEN_MARKERS = {"NPUUnflattenContiguousViews", "NPUDynamicMetaUnflattenViews"}


def covered(intervals):
    total = 0
    frontier = None
    for a,b in sorted(intervals):
        if frontier is None or a >= frontier:
            total += b-a
        elif b > frontier:
            total += b-frontier
        frontier = max(frontier or b,b)
    return total


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("leg",type=Path)
    args.add_argument("analysis",type=Path)
    args = args.parse_args()
    with gzip.open(args.leg/"trace_view.json.gz","rt",encoding="utf8") as f:
        events = json.load(f)
    if isinstance(events,dict):
        events = events["traceEvents"]
    pids = {e["args"]["name"]:e["pid"] for e in events if e["name"]=="process_name"}
    scopes, step_times, hardware, free, unflatten, syncs = [], {}, [], [], [], []
    for e in events:
        if e["ph"] != "X":
            continue
        s = Span.read(e)
        if e["pid"] == pids["Ascend Hardware"] and is_work(e.get("args",{}).get("Task Type","")):
            hardware.append(s)
        elif e["pid"] == pids["Overlap Analysis"] and s.name=="Free":
            free.append(s)
        elif s.name.startswith("ProfilerStep#"):
            step_times[int(s.name.split("#")[1])] = s
        elif e.get("cat") in {"cpu_op","enqueue","dequeue"} or s.name in UNFLATTEN_MARKERS | UNFLATTEN_OPS:
            scopes.append(s)
            if s.name in UNFLATTEN_OPS:
                unflatten.append(s)
        elif e["pid"] == pids["CANN"] and s.name.startswith("AscendCL@aclrtSynchronize"):
            syncs.append(s)
    del events
    report = json.loads((args.analysis/"submissions.json").read_bytes())
    rows, tails = [], []
    def intersect(samples, left, right):
        return covered(((max(s.start,left),min(s.end,right)) for s in samples if s.start < right and s.end > left))
    hardware.sort(key=lambda s:s.start)
    hw_starts = [s.start for s in hardware]
    for step in report["steps"]:
        number = step["step"]
        scope = step_times[number]
        meta = {k:step[k] for k in ("step","batch","seq_len")}
        for rank,gap in enumerate(step["largest_work_gaps"],1):
            left = scope.start + gap["step_offset_ns"]
            right = left + gap["gap_ns"]
            intersects = defaultdict(list)
            for s in scopes:
                if s.start<right and s.end>left:
                    intersects[s.tid,s.name].append((max(left,s.start),min(right,s.end)))
            for (tid,name),intervals in sorted(intersects.items(),key=lambda it:-covered(it[1])):
                total = covered(intervals)
                if total >= 200_000:
                    rows.append(dict(**meta,rank=rank, next_op=gap.get("op",""), tid=tid,
                                     scope=name, overlap_n=len(intervals),overlap_ms=total/1e6, gap_ms=gap["gap_ns"]/1e6))
        for s in unflatten + syncs:
            if scope.start <= s.start < scope.end:
                row=dict(**meta,scope=s.name,offset_ms=(s.start-scope.start)/1e6,
                         duration_ms=(s.end-s.start)/1e6,work_overlap_ms=intersect(hardware,s.start,s.end)/1e6,
                         free_overlap_ms=intersect(free,s.start,s.end)/1e6)
                if s.name in UNFLATTEN_OPS:
                    targets={"aten::narrow","aten::view","aten::slice","aten::as_strided"} | UNFLATTEN_MARKERS
                    samples={k:[] for k in sorted(targets)}
                    for x in scopes:
                        if x.tid==s.tid and s.start<=x.start<x.end<=s.end and x.name in samples:
                            samples[x.name].append(x.end-x.start)
                    for name,values in samples.items():
                        row[name]=len(values)
                        if len(values)>1:
                            row.update({f"{name}_{key}":value for key,value in stats(values).items() if key!="n"})
                    previous=max((x for x in hardware if x.end<=s.start),key=lambda x:x.end,default=None)
                    k=bisect_left(hw_starts,s.end)
                    after=hardware[k] if k<len(hardware) else None
                    if previous:
                        row.update(previous_task=previous.name,previous_stream=previous.args["Physic Stream Id"],
                                   previous_end_to_scope_us=(s.start-previous.end)/1000)
                    if after:
                        row.update(next_task=after.name,next_stream=after.args["Physic Stream Id"],
                                   scope_end_to_next_us=(after.start-s.end)/1000)
                tails.append(row)
        # This boundary excludes the next step if it falls outside the active profiler interval.
        i = bisect_left(hw_starts,scope.end)
        if i>0:
            last_work_end = max(s.end for s in hardware[:i])
            if scope.start <= last_work_end <= scope.end:
                tails.append(dict(**meta,scope="last_work_end_to_host_step_end",offset_ms=(last_work_end-scope.start)/1e6,
                                  duration_ms=(scope.end-last_work_end)/1e6,work_overlap_ms=0,
                                  free_overlap_ms=intersect(free,last_work_end,scope.end)/1e6))
                if i<len(hardware):
                    first = hardware[i].start
                    tails.append(dict(**meta,scope="last_work_end_to_next_work",offset_ms=(last_work_end-scope.start)/1e6,
                                      duration_ms=(first-last_work_end)/1e6,work_overlap_ms=0,
                                      free_overlap_ms=intersect(free,last_work_end,first)/1e6))
    write_csv(args.analysis/"largest_gap_cpu_anchors.csv",rows)
    write_csv(args.analysis/"drain_and_tail.csv",tails)


if __name__=="__main__":
    main()
