"""Find adjacent SQ entries from one eager request that were sent before the prior device task began."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys
import tarfile

cli=argparse.ArgumentParser()
cli.add_argument("analyzer",type=Path)
args=cli.parse_args()
sys.path.insert(0,str(args.analyzer.resolve().parent))
from analyze_dynamic_trace import Lanes,Span

root=Path(__file__).resolve().parent
with gzip.open(root/"calls.csv.gz","rt",newline="") as fp:
    calls=list(csv.DictReader(fp))
with tarfile.open(root/"evidence.tgz") as arc:
    member,=[m for m in arc.getmembers() if m.name.endswith("/02-B/trace_view.json.gz")]
    with gzip.open(arc.extractfile(member)) as stream:
        trace=json.load(stream)
events=trace if isinstance(trace,list) else trace["traceEvents"]
queues=Lanes([Span.read(e) for e in events if e.get("cat")=="dequeue" and e["ph"]=="X"])
limits=json.loads((root/"hal-bounds.json").read_bytes())
report={}
for step in ("4","5"):
    group=sorted((r for r in calls if r["step"]==step),key=lambda r:int(r["start_ns"]))
    previous={}
    counts=Counter()
    totals=Counter()
    per_op=defaultdict(Counter)
    # key by queue id as well as TID. A single named operator can recur 1,024 times.
    for row in group:
        sq=row["sq_id"]
        last=previous.get(sq)
        previous[sq]=row
        probe=Span(int(row["start_ns"]),int(row["end_ns"]),int(row["tid"]),"HAL",{})
        req=queues.parent(probe)
        row["request"]=(req.tid,req.args["correlation_id"]) if req else None
        if not last or not row["gap_ns"] or not last["device_end_ns"]:
            continue
        if (row["stream"]!=last["stream"] or row["request"] is None or row["request"]!=last["request"]
            or int(row["gap_left_ns"])!=int(last["device_end_ns"])
            or (int(row["driver_position"])-int(last["driver_position"]))%2048!=1):
            continue
        key=row["op"]
        gap=int(row["gap_ns"])
        for label,offset in (("earliest",limits[step]["lower_ns"]),("nominal",0),("latest",limits[step]["upper_ns"])):
            if int(row["end_ns"])<=int(last["device_start_ns"])+offset:
                counts[label]+=1
                totals[label]+=gap
                per_op[key][label]+=gap
        counts["eligible"]+=1
        totals["eligible"]+=gap
        per_op[key]["eligible"]+=gap
    report[step]=dict(counts=counts,total_gap_ms={k:v/1e6 for k,v in totals.items()},
        by_dequeue_ms={k:{j:t/1e6 for j,t in v.items()} for k,v in sorted(per_op.items())})
(root/"already-sent.json").write_text(json.dumps(report,indent=2))
