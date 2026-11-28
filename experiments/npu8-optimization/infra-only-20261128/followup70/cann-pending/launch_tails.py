"""Match each physical work item to one actual CANN submission and measure API tails and adjoining idle."""
import argparse
from collections import Counter,defaultdict
import gzip
import json
import math
from pathlib import Path
import statistics
import sys


def summarize(rows):
    vals=sorted((b-a)/1000 for a,b,_,_ in rows)
    gap=sum(r-l for _,_,l,r in rows)
    direct=sum(max(0,min(b,r)-max(a,l)) for a,b,l,r in rows)
    return dict(n=len(vals),median_us=statistics.median(vals),p95_us=vals[math.ceil(.95*len(vals))-1],
                p99_us=vals[math.ceil(.99*len(vals))-1],max_us=vals[-1],direct_gap_ms=gap/1e6,
                nominal_intersection_with_own_acl_ms=direct/1e6)


def run(path):
    from analyze_dynamic_trace import Span,Lanes,is_work,ns
    with gzip.open(path,mode="rt",encoding="utf-8-sig") as f:
        v=json.load(f)
    rows=v if isinstance(v,list) else v["traceEvents"]
    pids={e["args"]["name"]:e["pid"] for e in rows if e["name"]=="process_name"}
    src,dst,anchors=defaultdict(list),defaultdict(list),defaultdict(list)
    steps,physical,acl={},{},[]
    for e in rows:
        n,pid,ph=e["name"],e["pid"],e["ph"]
        if e.get("cat")=="HostToDevice":
            if ph=="s" and pid==pids["CANN"]:src[e["id"]].append((e["tid"],ns(e["ts"])))
            if ph=="f" and pid==pids["Ascend Hardware"]:dst[e["tid"],ns(e["ts"])].append(e["id"])
        if ph!="X":continue
        s=Span.read(e)
        if n.startswith("ProfilerStep#"):steps[int(n.split("#")[-1])]=s
        if pid==pids["CANN"]:
            anchors[s.tid,s.start,s.args.get("connection_id")].append(s)
            if n.startswith(("AscendCL@aclrtLaunchKernel","AscendCL@aclrtMemcpy")):acl.append(s)
        if pid==pids["Ascend Hardware"] and is_work(s.args.get("Task Type","")):
            physical.setdefault(s.start,[]).append(s)
    ix=Lanes(acl)
    result={}
    frontier=None
    by_step=defaultdict(lambda:defaultdict(list))
    matched=defaultdict(Counter)
    addresses=defaultdict(Counter)
    for tick,batch in sorted(physical.items()):
        left=frontier
        frontier=max(frontier or 0,max(t.end for t in batch))
        step=next((k for k,sc in steps.items() if sc.start<=tick<sc.end),None)
        if step is None:continue
        for t in batch:
            matched[step]["physical"]+=1
            ids=dst[t.tid,t.start]
            if len(ids)!=1 or len(src[ids[0]])!=1:continue
            tid,start=src[ids[0]][0]
            candidates=anchors[tid,start,t.args.get("connection_id")]
            if len(candidates)!=1:continue
            apis=ix.children(candidates[0])
            if len(apis)!=1:continue
            now=apis[0]
            if not(steps[step].start<=now.start and now.end<=steps[step].end):continue
            addresses[step][now.tid,now.start,now.end]+=1
            matched[step]["linked"]+=1
            l=left if left is not None and steps[step].start<=left<tick and len(batch)==1 else tick
            row=(now.start,now.end,l,tick)
            for key in ("all","api:"+now.name,"stream:"+str(t.args.get("Physic Stream Id")),"task:"+t.name):
                by_step[step][key].append(row)
    for step,groups in by_step.items():
        result[step]=dict(**matched[step],distinct_api_calls=len(addresses[step]),
                          multiwork_api_calls=sum(n>1 for n in addresses[step].values()),
                          by={key:summarize(v) for key,v in groups.items()})
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run",type=Path)
    p.add_argument("analysis_owner",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    sys.path.insert(0,str(a.analysis_owner.parent))
    report={d.name[-1]:run(d/"trace_view.json.gz") for d in sorted(a.run.glob("0?-?"))}
    if set(report)!={"A","B"}:raise ValueError(report.keys())
    a.out.write_text(json.dumps(report,indent=2)+"\n")
