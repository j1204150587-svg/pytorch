"""Count the actual host index views between side-table arithmetic and each embedding."""
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import gzip
import json
from pathlib import Path
from trace_budget import root,read,Span,Lanes

def analyze():
    analysis=root/"embedding/analysis/emb-fwd/B"
    leg=root/"evidence/raw/20261128T043752Z-ab-l1-7m6ais34/02-B"
    with gzip.open(leg/"trace_view.json.gz") as f:
        z=json.load(f)
    events=z if isinstance(z,list) else z["traceEvents"]
    names={"aten::slice","aten::select","aten::add","aten::mean","aten::embedding","Enqueue@aclnnEmbedding"}
    spans=[Span.read(e) for e in events if e.get("ph")=="X" and (e["name"] in names or e["name"].startswith("ProfilerStep#"))]
    schemes={s.name.split("#")[-1]:s.start for s in spans if s.name.startswith("ProfilerStep#")}
    emb=Lanes([s for s in spans if s.name=="aten::embedding"])
    q={str(s.args["correlation_id"]):s for s in spans if s.name=="Enqueue@aclnnEmbedding"}
    ids={(t["step"],t["queue_id"]) for t in read(analysis/"work_tasks.csv.gz") if t["task"]=="aclnnEmbedding_AsStridedAiCore_AsStrided"}
    views=Lanes([s for s in spans if s.name in {"aten::slice","aten::select"}])
    fences=Lanes([s for s in spans if s.name in {"aten::add","aten::mean","aten::embedding"}])
    raw=read(analysis/"device_gaps.csv.gz")
    result={}
    for step in sorted({s for s,_ in ids}):
        n=Counter()
        gaps=sorted((schemes[step]+int(x["step_offset_ns"]),int(x["gap_ns"]),x.get("op") or "unlinked") for x in raw if x["step"]==step)
        starts=[t for t,d,o in gaps]
        budget=defaultdict(Counter)
        seen=set()
        for _,qid in (k for k in ids if k[0]==step):
            owner=emb.parent(q[qid])
            i=bisect_left(views.starts[owner.tid],owner.start)
            f=bisect_left(fences.starts[owner.tid],owner.start)-1
            start=bisect_left(views.starts[owner.tid],fences.lanes[owner.tid][f].end)
            parts=views.lanes[owner.tid][start:i]
            if not parts or parts[-1].name!="aten::select" or parts[-1].end>owner.start:
                raise ValueError((step,qid,[p.name for p in parts]))
            for p in parts:
                if (p.tid,p.start) in seen:
                    raise ValueError("nonunique view operation")
                seen.add((p.tid,p.start))
                n[p.name]+=1
                stat=budget[p.name]
                stat["cpu"]+=p.end-p.start
                j=max(0,bisect_right(starts,p.start)-1)
                while j<len(gaps) and gaps[j][0]<p.end:
                    t,d,op=gaps[j]
                    overlap=max(0,min(t+d,p.end)-max(t,p.start))
                    stat["all_device_gaps"]+=overlap
                    stat[op]+=overlap
                    j+=1
        result[step]={"counts":n,"intersection_ms":{k:{col:round(x/1e6,6) for col,x in c.items() if x} for k,c in budget.items()}}
    return result

if __name__=="__main__":
    Path(__file__).with_name("index-view-budget.json").write_text(json.dumps(analyze(),indent=2)+"\n")
