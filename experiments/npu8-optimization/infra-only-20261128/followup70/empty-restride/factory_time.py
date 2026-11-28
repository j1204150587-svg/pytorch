"""Intersect the real empty_tensor CPU scopes with gaps across all physical work streams."""
from bisect import bisect_right
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

leg, analyzer, out = map(Path, sys.argv[1:])
sys.path.insert(0,str(analyzer.parent))
from analyze_dynamic_trace import Span, is_work, stats
with gzip.open(leg/"trace_view.json.gz") as f: data=json.load(f)
events=data if isinstance(data,list) else data["traceEvents"]
device=next(e["pid"] for e in events if e["name"]=="process_name" and e["args"]["name"]=="Ascend Hardware")
tasks, factories, steps = [], [], {}
for event in events:
    if event["ph"]!="X": continue
    if event["pid"]==device and is_work(event.get("args",{}).get("Task Type","")):
        tasks.append(Span.read(event))
    if event["name"]=="empty_tensor": factories.append(Span.read(event))
    if event["name"].startswith("ProfilerStep#"):
        steps[int(event["name"].split("#")[1])]=Span.read(event)
def union(pairs):
    result=[]
    for a,b in sorted(pairs):
        if b<=a: continue
        if result and result[-1][1]>=a: result[-1][1]=max(b,result[-1][1])
        else: result.append([a,b])
    return result
work=union((s.start,s.end) for s in tasks)
holes=[(a[1],b[0]) for a,b in zip(work,work[1:]) if a[1]<b[0]]
result={}
for n,step in sorted(steps.items()):
    fs=[s for s in factories if step.start<=s.start<=s.end<=step.end]
    u=union((s.start,s.end) for s in fs)
    starts=[a for a,b in u]
    overlap=0
    for a,b in holes:
        if not step.start<=a<b<=step.end: continue
        i=max(0,bisect_right(starts,a)-1)
        while i<len(u) and u[i][0]<b:
            overlap+=max(0,min(b,u[i][1])-max(a,u[i][0]))
            i+=1
    result[n]=dict(factory=stats([s.end-s.start for s in fs]),
                   union_ms=sum(b-a for a,b in u)/1e6, all_stream_gap_overlap_ms=overlap/1e6)
out.write_text(json.dumps(result,indent=2)+"\n")
