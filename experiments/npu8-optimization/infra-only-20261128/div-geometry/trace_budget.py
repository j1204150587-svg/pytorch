"""Intersect actual side-table mean-backward CPU intervals with the all-stream gap before each targeted Divs."""
import bisect
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parents[1]
base=root.parent
sys.path.insert(0,str(base/"dynamic-infra-20261127/analysis"))
from analyze_dynamic_trace import Span,Lanes


def read(f):
    opener=gzip.open if f.suffix==".gz" else open
    with opener(f,"rt",encoding="utf-8-sig") as s:
        return list(csv.DictReader(s))

def exclusive_cpu(events,tid):
    edges=defaultdict(list)
    for i,e in enumerate(events):
        if e.get("ph")!="X" or e.get("tid")!=tid or e.get("cat")!="cpu_op":
            continue
        span=Span.read(e)
        if span.end>span.start:
            tag=(span.start,-span.end,i,span.name)
            edges[span.start].append((True,tag))
            edges[span.end].append((False,tag))
    active=set()
    boundaries=sorted(edges)
    labels=[]
    for t in boundaries:
        for opening,tag in sorted(edges[t]):
            if opening:
                active.add(tag)
            else:
                active.discard(tag)
        labels.append(max(active)[3] if active else "no CPU scope on worker")
    return boundaries,labels

def partition(lane,left,right,totals):
    times,labels=lane
    i=bisect.bisect_right(times,left)-1
    while 0<=i<len(labels)-1 and times[i]<right:
        totals[labels[i]]+=max(0,min(right,times[i+1])-max(left,times[i]))
        i+=1


def run():
    leg=root/"evidence/raw/20261128T043752Z-ab-l1-7m6ais34/02-B"
    analysis=root/"embedding/analysis/emb-fwd/B"
    profile_leg=Path(chr(92)*2+"?"+chr(92)+str(leg)) if sys.platform=="win32" else leg
    kernels=read(next(profile_leg.rglob("kernel_details.csv")))
    ids={(k["Step Id"],k["Stream ID"],k["Task ID"]) for k in kernels
         if k["Name"]=="aclnnDivs_BroadcastToAiCore_BroadcastTo" and ",1,64;" in k["Input Shapes"]}
    selected={(t["step"],t["queue_id"]) for t in read(analysis/"work_tasks.csv.gz") if (t["step"],t["stream"],t["task_id"]) in ids}
    with gzip.open(leg/"trace_view.json.gz") as f:
        z=json.load(f)
    events=z if isinstance(z,list) else z["traceEvents"]
    names={"MeanBackward1","aten::unsqueeze","aten::expand","aten::as_strided","aten::div","empty_tensor","Enqueue@aclnnDivs"}
    spans=[Span.read(e) for e in events if e.get("ph")=="X" and (e["name"] in names or e["name"].startswith("ProfilerStep#"))]
    steps={e.name.split("#")[-1]:e.start for e in spans if e.name.startswith("ProfilerStep#")}
    indices={n:Lanes([e for e in spans if e.name==n]) for n in names}
    enq={str(e.args["correlation_id"]):e for e in spans if e.name=="Enqueue@aclnnDivs"}
    worker_tids={enq[q].tid for _,q in selected}
    workers={t:exclusive_cpu(events,t) for t in worker_tids}
    preceding=defaultdict(lambda:defaultdict(int))
    gaps=defaultdict(list)
    for x in read(analysis/"device_gaps.csv.gz"):
        k=x["step"],x["queue_id"]
        if k in selected:
            left=steps[k[0]]+int(x["step_offset_ns"])
            gaps[k].append((left,left+int(x["gap_ns"])))
    rows=defaultdict(list)
    for step,qid in selected:
        q=enq[qid]
        m=indices["MeanBackward1"].parent(q)
        div=indices["aten::div"].parent(q)
        if not m or not div:
            raise ValueError((step,qid))
        g=gaps[step,qid]
        for left,right in g:
            if left<m.start:
                partition(workers[q.tid],left,min(right,m.start),preceding[step])
        def obs(ranges):
            return {"cpu_ns":sum(b-a for a,b in ranges),"overlap_with_next_div_gap_ns":sum(max(0,min(b,r)-max(a,l)) for a,b in ranges for l,r in g),"n":len(ranges)}
        out={"mean_entry_to_div":obs([(m.start,div.start)]),"div_entry_to_enqueue":obs([(div.start,q.start)]),
             "gap_full":sum(r-l for l,r in g),
             "gap_before_mean_entry":sum(max(0,min(m.start,r)-l) for l,r in g)}
        for n in ("aten::unsqueeze","aten::expand","aten::as_strided","empty_tensor"):
            out[n]=obs([(e.start,e.end) for e in indices[n].children(m) if e.end<=q.start])
        rows[step].append(out)
    return {s:{"calls":len(rr),"gap_full_ms":sum(r["gap_full"] for r in rr)/1e6,
               "gap_before_mean_entry_ms":sum(r["gap_before_mean_entry"] for r in rr)/1e6,
               "before_mean_worker_exclusive_ms":{k:v/1e6 for k,v in sorted(preceding[s].items(),key=lambda kv:-kv[1])},
               "intervals":{k:{"count":sum(r[k]["n"] for r in rr),"cpu_ms":sum(r[k]["cpu_ns"] for r in rr)/1e6,
                         "overlap_ms":sum(r[k]["overlap_with_next_div_gap_ns"] for r in rr)/1e6}
                   for k in rr[0] if isinstance(rr[0][k],dict)}} for s,rr in sorted(rows.items())}


if __name__=="__main__":
    Path(__file__).with_name("trace-budget.json").write_text(json.dumps(run(),indent=2)+"\n")
