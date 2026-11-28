"""Attribute Level1 submit calls to each of the four synchronized microprobe markers."""
from collections import Counter
from decimal import Decimal
import gzip
import json
from pathlib import Path
import statistics
import sys


def analyze(file):
    with gzip.open(file,"rt",encoding="utf-8") as f:
        data=json.load(f)
    raw = data if isinstance(data,list) else data["traceEvents"]
    hardware, = {e["pid"] for e in raw if e["name"]=="process_name" and e["args"]["name"]=="Ascend Hardware"}
    rows = []
    for e in raw:
        if e.get("ph")=="X":
            start=int(Decimal(str(e["ts"]))*1000)
            dur=int(Decimal(str(e["dur"]))*1000)
            rows.append(dict(e,start=start,end=start+dur,duration=dur))
    def inside(e,owner):
        return owner["start"]<=e["start"] and e["end"]<=owner["end"]
    def group(items):
        return {k:dict(count=len(u),mean_us=round(statistics.mean(u)/1000,3),
                       median_us=round(statistics.median(u)/1000,3),max_us=round(max(u)/1000,3))
                for k in items[0] if (u:=[x[k] for x in items])}
    result=[]
    for mark in sorted((e for e in rows if e["name"].startswith("DivsProbe_")),key=lambda x:x["start"]):
        local=[e for e in rows if inside(e,mark)]
        queue=[e for e in local if e["name"].startswith("Enqueue@")]
        deq={e["args"]["correlation_id"]:e for e in local if e["name"].startswith("Dequeue@")}
        tasks=[e for e in local if e["pid"]==hardware and e["args"].get("Task Type")=="AI_VECTOR_CORE"]
        assert len(queue)==len(deq)==8 and {t["args"]["Physic Stream Id"] for t in tasks}=={47}
        calls=[]
        for q in queue:
            p,=[e for e in local if e["name"]=="aten::div" and e["tid"]==q["tid"] and inside(q,e)]
            d=deq[q["args"]["correlation_id"]]
            launches=[e for e in local if e["name"].startswith("AscendCL@aclrtLaunchKernel") and e["tid"]==d["tid"] and inside(e,d)]
            calls.append(dict(producer_ns=q["start"]-p["start"],enqueue_ns=q["duration"],
                              wait_ns=d["start"]-q["end"],consumer_ns=d["duration"],
                              launch_total_ns=sum(e["duration"] for e in launches),acl_launches=len(launches)))
        api_sum=sum(c.pop("acl_launches") for c in calls)
        assert api_sum==len(tasks)
        result.append(dict(marker=mark["name"],requests=len(calls),launches=api_sum,physical_stream=47,
                           work=dict(Counter(e["name"] for e in tasks)),rows=calls,statistics=group(calls),
                           views={name:round(statistics.mean(v)/1000,3) for name in ("aten::narrow","aten::slice","aten::as_strided")
                                  if (v:=[e["duration"] for e in local if e["name"]==name])}))
    assert len(result)==4
    return result


if __name__=="__main__":
    path=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).with_name("divs-trace.json.gz")
    output=Path(sys.argv[2]) if len(sys.argv)>2 else path.with_name("micro-attribution.json")
    output.write_text(json.dumps(analyze(path),indent=2)+"\n",encoding="utf-8")
