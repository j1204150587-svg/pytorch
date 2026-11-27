"""Split recorded queue wait exposed by work gaps against earlier consumer service."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
from analyze_dynamic_trace import Span, stats, write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg",type=Path)
    parser.add_argument("analysis",type=Path)
    args=parser.parse_args()
    with gzip.open(args.leg/"trace_view.json.gz","rt",encoding="utf-8") as f:
        events=json.load(f)
    if isinstance(events,dict):
        events=events["traceEvents"]
    eq,dq,steps,records={},{},{},defaultdict(list)
    for e in events:
        if e["ph"]!="X":
            continue
        if e["name"].startswith("ProfilerStep#"):
            s=Span.read(e)
            steps[int(s.name.split("#")[-1])]=s.start
        elif e.get("cat")=="dequeue":
            s=Span.read(e)
            dq[s.args["correlation_id"]]=s
            records[s.tid].append(s)
        elif e.get("cat")=="enqueue":
            s=Span.read(e)
            eq[s.args["correlation_id"]]=s
    del events
    starts={}
    for tid,lane in records.items():
        lane.sort(key=lambda x:x.start)
        if any(s.end>t.start for s,t in zip(lane,lane[1:])):
            raise ValueError("overlapping consumer records")
        starts[tid]=[s.start for s in lane]
    out=defaultdict(lambda:defaultdict(list))
    with gzip.open(args.analysis/"device_gaps.csv.gz","rt",encoding="utf8",newline="") as f:
        for r in csv.DictReader(f):
            if r["state"]!="linked":
                continue
            step=int(r["step"])
            q=int(r["queue_id"])
            en,de=eq[q],dq[q]
            a=steps[step]+int(r["step_offset_ns"])
            b=a+int(r["gap_ns"])
            lo=max(a,min(de.start,en.end))
            hi=min(b,de.start)
            duration=max(0,hi-lo)
            if duration!=int(r["visible_queue_wait_ns"]):
                raise ValueError("queue boundary differs from attribution")
            covered=0
            if duration:
                i=max(0,bisect_right(starts[de.tid],lo)-1)
                while i<len(records[de.tid]):
                    s=records[de.tid][i]
                    if s.start>=hi:
                        break
                    covered+=max(0,min(s.end,hi)-max(s.start,lo))
                    i+=1
            for op,role in (("ALL","ALL"),(r["op"],r["role"])):
                group=out[step,op,role]
                group["visible_queue_wait"].append(duration)
                group["during_preceding_dequeue"].append(covered)
                group["outside_recorded_service"].append(duration-covered)
    rows=[dict(step=step,op=op,role=role,metric=metric,positive_n=sum(x>0 for x in values),
               **stats(values)) for (step,op,role),group in sorted(out.items()) for metric,values in group.items()]
    write_csv(args.analysis/"queue_wait_overlap.csv",rows)


if __name__=="__main__":
    main()
