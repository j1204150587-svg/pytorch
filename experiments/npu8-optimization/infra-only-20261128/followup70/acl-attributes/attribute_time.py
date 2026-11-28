"""Align stream attributes and per-thread resources with dynamic HSTU all-stream device gaps."""
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
study=here.parents[1]
sys.path.insert(0,str(study.parent/"dynamic-infra-20261127/analysis"))
from analyze_dynamic_trace import ns

path=study/"evidence/raw/20261128T043752Z-ab-l1-7m6ais34/02-B/trace_view.json.gz"
gpath=study/"embedding/analysis/emb-fwd/B/device_gaps.csv.gz"
names=("AscendCL@aclrtGetStreamAttribute","AscendCL@aclrtGetResInCurrentThread")
with gzip.open(path) as f:
    data=json.load(f)
events=data if isinstance(data,list) else data["traceEvents"]
steps={}
dequeues=defaultdict(list)
attrs=defaultdict(list)
for e in events:
    if e.get("ph")!="X":
        continue
    label=e["name"]
    if label.startswith("ProfilerStep#"):
        start=ns(e["ts"])
        steps[label.split("#")[-1]]=(start,start+ns(e["dur"]))
    elif e.get("cat")=="dequeue":
        a=ns(e["ts"])
        dequeues[e["tid"]].append((a,a+ns(e["dur"]),str(e["args"]["correlation_id"])))
    elif label in names:
        a=ns(e["ts"])
        attrs[label].append((a,a+ns(e["dur"]),e["tid"]))
with gzip.open(gpath,"rt",encoding="utf-8-sig") as fp:
    raw=list(csv.DictReader(fp))
def merge(values):
    result=[]
    for a,b in sorted(values):
        if result and a<=result[-1][1]:
            result[-1]=(result[-1][0],max(b,result[-1][1]))
        elif b>a:
            result.append((a,b))
    return result
def intersect(a,b):
    i=j=value=0
    while i<len(a) and j<len(b):
        value+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<b[j][1]:
            i+=1
        else:
            j+=1
    return value
def quantile(seq,p):
    return sorted(seq)[(len(seq)*p+99)//100-1]/1000
de={tid:sorted(rows) for tid,rows in dequeues.items()}
starts={tid:[r[0] for r in rows] for tid,rows in de.items()}
results={}
for st,(start,end) in steps.items():
    gg=[r for r in raw if r["step"]==st]
    gaps=defaultdict(list)
    for row in gg:
        a=start+int(row["step_offset_ns"])
        gaps[row["queue_id"]].append((a,a+int(row["gap_ns"])))
    full=merge(span for spans in gaps.values() for span in spans)
    results[st]={}
    for name in names:
        times=[(a,b,tid) for a,b,tid in attrs[name] if start<=a and b<=end]
        union=merge((a,b) for a,b,_ in times)
        length=[b-a for a,b,_ in times]
        same=inside=0
        if name==names[0]:
            for a,b,tid in times:
                i=bisect_right(starts.get(tid,[]),a)-1
                if i>=0:
                    x,y,queue=de[tid][i]
                    if x<=a and b<=y:
                        inside+=1
                        same+=intersect([(a,b)],sorted(gaps[queue]))
        results[st][name]=dict(calls=len(times),by_tid=dict(Counter(t for _,_,t in times)),sum_ms=sum(length)/1e6,
            union_ms=sum(b-a for a,b in union)/1e6,union_gap_intersection_ms=intersect(union,full)/1e6,
            p50_us=quantile(length,50),p95_us=quantile(length,95),p99_us=quantile(length,99),max_us=max(length)/1000)
        if name==names[0]:
            results[st][name].update(in_dequeue=inside,own_successor_gap_intersection_ms=same/1e6)
(here/"windows.json").write_text(json.dumps(dict(trace=hashlib.sha256(path.read_bytes()).hexdigest(),
    gap_csv=hashlib.sha256(gpath.read_bytes()).hexdigest(),steps=results),indent=2)+"\n")
