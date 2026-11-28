"""Attribute each scored forward/backward to queue service, CANN calls and physical work."""
import argparse
from collections import Counter
from decimal import Decimal
import gzip
import json
from pathlib import Path


def span(row):
    start=int(Decimal(str(row["ts"]))*1000)
    return start,start+int(Decimal(str(row["dur"]))*1000)


def collect(path):
    with (gzip.open if path.suffix==".gz" else open)(path,"rt",encoding="utf-8-sig") as fh:
        doc=json.load(fh)
    allrows=doc if isinstance(doc,list) else doc["traceEvents"]
    hw,={r["pid"] for r in allrows if r["name"]=="process_name" and r["args"]["name"]=="Ascend Hardware"}
    rows=[r for r in allrows if r["ph"]=="X"]
    marks=sorted((r for r in rows if r["name"].startswith("SCORELOCK_")),key=span)
    if len(marks)!=8:raise ValueError("Expected four S/P pairs")
    deq={str(r["args"]["correlation_id"]):r for r in rows if r.get("cat")=="dequeue"}
    out=[]
    for i,mark in enumerate(marks):
        a,b=span(mark)
        next_mark=span(marks[i+1])[0] if i+1<len(marks) else max(span(x)[1] for x in rows)
        calls=[]
        for enq in rows:
            if enq.get("cat")!="enqueue" or "causal_score_" not in enq["name"]:continue
            es,ee=span(enq)
            if not a<=es<ee<=b:continue
            cid=str(enq["args"]["correlation_id"])
            consumer=deq[cid]
            ds,dd=span(consumer)
            sub=[r for r in rows if r["tid"]==consumer["tid"] and ds<=span(r)[0] and span(r)[1]<=dd
                 and any(k in r["name"].lower() for k in ("launchkernel","rtmemcpy","kernellaunch"))]
            calls.append({"op":enq["name"],"correlation":cid,"queue_us":(ds-ee)/1000,
                          "consumer_us":(dd-ds)/1000,
                          "runtime":[[r["name"],float(r["dur"])] for r in sub]})
        physical=sorted((r for r in rows if r["pid"]==hw and a<=span(r)[0]<next_mark and
                         "Task Id" in r.get("args",{})),key=span)
        details=[{"name":r["name"],"stream":r["args"]["Physic Stream Id"],"type":r["args"]["Task Type"],
                  "task":r["args"]["Task Id"]} for r in physical]
        payload=[v for v in details if "causal_score" in v["name"] or "MEMCPY" in v["type"].upper()]
        out.append({"marker":mark["name"],"host_dispatch_us":(b-a)/1000,"calls":calls,
                    "score_or_memcpy":payload,"whole_drained_work":details})
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args();a.output.write_text(json.dumps(collect(a.trace),indent=2)+"\n")
