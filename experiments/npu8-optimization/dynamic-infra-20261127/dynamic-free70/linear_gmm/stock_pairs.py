"""Match the two MM enqueues in each stock AddmmBackward0 to actual device work."""
import argparse
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path
from collections import Counter


def table(path):
    with gzip.open(path,"rt",encoding="utf8") as f:return list(csv.DictReader(f))


def interval(e):
    a=Decimal(str(e["ts"]))
    return a,a+Decimal(str(e["dur"]))


def inside(e,parent):
    a,b=interval(e)
    l,r=interval(parent)
    return l<=a<=b<=r


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("trace",type=Path)
    arg.add_argument("analysis",type=Path)
    arg.add_argument("output",type=Path)
    args=arg.parse_args()
    with gzip.open(args.trace,"rt",encoding="utf8") as f:raw=json.load(f)
    events=raw if isinstance(raw,list) else raw["traceEvents"]
    scopes=[v for v in events if v["ph"]=="X" and v["name"]=="AddmmBackward0"]
    enq=[v for v in events if v["ph"]=="X" and v["name"]=="Enqueue@aclnnMm"]
    q={r["queue_id"]:r for r in table(args.analysis/"queue_calls.csv.gz")}
    work=table(args.analysis/"work_tasks.csv.gz")
    gaps=table(args.analysis/"device_gaps.csv.gz")
    report=[]
    for step in sorted({r["step"] for r in work},key=int):
        pairs=[]
        for s in sorted(scopes,key=interval):
            ids=[str(e["args"]["correlation_id"]) for e in sorted(enq,key=interval) if s["tid"]==e["tid"] and inside(e,s)]
            if not ids or q[ids[0]]["step"]!=step: continue
            assert len(ids)==2
            assert all(q[i]["aten"]=="aten::mm" and q[i]["step"]==step for i in ids)
            ts=[r for r in work if r["queue_id"] in ids]
            gs=[r for r in gaps if r["queue_id"] in ids]
            pairs.append(dict(ids=ids,kernels=[[r["stream"],int(r["stream_work_slot"]),r["task"]] for r in ts],
                requested=[q[i]["acl_api_sequence"] for i in ids],
                trace_scope_us=s["dur"],copy_of_inputs_in_profile=None,
                preceding_all_device_gap_ms=sum(int(r["gap_ns"]) for r in gs)/1e6))
        report.append(dict(step=int(step),pairs=pairs,all_pair_gap_ms=sum(p["preceding_all_device_gap_ms"] for p in pairs),
                           all_addmm_backward_scopes_ms=sum(float(p["trace_scope_us"]) for p in pairs)/1000))
    args.output.write_text(json.dumps(report,indent=2)+"\n")
