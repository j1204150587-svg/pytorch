"""Compute the union of producer empty_tensor intervals intersected with real all-stream gaps."""
from bisect import bisect_right
from collections import Counter
import gzip
import json
from pathlib import Path
from trace_budget import root,Span,read


def main():
    analysis=root/"embedding/analysis/emb-fwd/B"
    leg=root/"evidence/raw/20261128T043752Z-ab-l1-7m6ais34/02-B"
    with gzip.open(leg/"trace_view.json.gz") as f:
        z=json.load(f)
    events=z if isinstance(z,list) else z["traceEvents"]
    steps={e["name"].split("#")[-1]:Span.read(e) for e in events if e.get("ph")=="X" and e["name"].startswith("ProfilerStep#")}
    alloc=[Span.read(e) for e in events if e.get("ph")=="X" and e["name"]=="empty_tensor"]
    gaps=read(analysis/"device_gaps.csv.gz")
    data={}
    for step,owner in steps.items():
        a=[e for e in alloc if owner.start<=e.start<owner.end]
        if not a:
            continue
        union=[]
        for e in sorted(a,key=lambda x:x.start):
            if union and union[-1][1]>=e.start:
                union[-1][1]=max(union[-1][1],e.end)
            else:
                union.append([e.start,e.end])
        starts=[x[0] for x in union]
        overlaps=Counter()
        for row in (r for r in gaps if r["step"]==step):
            left=owner.start+int(row["step_offset_ns"])
            right=left+int(row["gap_ns"])
            i=max(0,bisect_right(starts,left)-1)
            while i<len(union) and union[i][0]<right:
                overlaps[row["op"] or "unlinked"]+=max(0,min(union[i][1],right)-max(union[i][0],left))
                i+=1
        data[step]={"calls":len(a),"on_step_main":sum(e.tid==owner.tid for e in a),
                    "union_cpu_ms":sum(b-a for a,b in union)/1e6,"intersection_ms":sum(overlaps.values())/1e6,
                    "by_next_request_ms":{c:ns/1e6 for c,ns in overlaps.most_common() if ns}}
    Path(__file__).with_name("empty-budget.json").write_text(json.dumps(data,indent=2)+"\n")


if __name__=="__main__":
    main()
