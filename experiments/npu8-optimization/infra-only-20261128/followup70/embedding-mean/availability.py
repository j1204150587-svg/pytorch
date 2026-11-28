"""Check whether adjacent eager embedding and mean requests are already available to the consumer."""
from collections import Counter,defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import statistics

here=Path(__file__).resolve().parent
study=here.parents[1]
base=study/"embedding/analysis/emb-fwd/B"
def rows(name):
    with gzip.open(base/name,"rt",encoding="utf-8-sig",newline="") as fp:
        return list(csv.DictReader(fp))
work=rows("work_tasks.csv.gz")
queue=rows("queue_calls.csv.gz")
tasks=defaultdict(list)
for r in work:
    tasks[r["step"],r["queue_id"]].append(r)
trace=study/"evidence/raw/20261128T043752Z-ab-l1-7m6ais34/02-B/trace_view.json.gz"
with gzip.open(trace) as fp:
    raw=json.load(fp)
events=raw if isinstance(raw,list) else raw["traceEvents"]
en,de={},{}
def ns(t):
    whole,_,dec=str(t).partition(".")
    return int(whole)*1000+int((dec+"000")[:3])
for e in events:
    if e.get("ph")!="X" or e.get("cat") not in ("enqueue","dequeue"):
        continue
    key=str(e["args"]["correlation_id"])
    t=ns(e["ts"])
    (en if e["cat"]=="enqueue" else de)[key]=(t,t+ns(e["dur"]),e["tid"])
outputs={}
full=[]
for step in ("4","5"):
    q=sorted([r for r in queue if r["step"]==step and r["role"]=="main"],
             key=lambda r:en[r["queue_id"]][0])
    group=[]
    for i,r in enumerate(q[:-1]):
        a=r["queue_id"]
        physical=tasks[step,a]
        if not any(t["task"]=="aclnnEmbedding_AsStridedAiCore_AsStrided" for t in physical):
            continue
        b=q[i+1]["queue_id"]
        assert (r["op"],q[i+1]["op"])==("aclnnEmbedding","aclnnMean")
        assert {t["stream"] for t in physical+tasks[step,b]}=={"47"}
        assert len(physical)==2 and len(tasks[step,b])==1
        e_first,e_last,_=en[b]
        d_first,d_last,_=de[a]
        group.append(dict(step=int(step),batch=int(r["batch"]),length=int(r["seq_len"]),
                          embedding=a,mean=b,mean_enqueue_start_minus_embedding_dequeue_start_ns=e_first-d_first,
                          mean_enqueue_end_minus_embedding_dequeue_start_ns=e_last-d_first,
                          mean_enqueue_start_minus_embedding_dequeue_end_ns=e_first-d_last))
    assert len(group)==1024
    full+=group
    deadlines={}
    for budget in (0,1,2,5,10,20,50):
        keys=[k for k in group[0] if "minus_embedding_dequeue_start_ns" in k]
        deadlines[str(budget)]={key:sum(r[key]<=1000*budget for r in group) for key in keys}
    after=[r["mean_enqueue_end_minus_embedding_dequeue_start_ns"] for r in group]
    outputs[step]=dict(shape=[group[0]["batch"],group[0]["length"]],pairs=len(group),matched_launches=3*len(group),
        deadlines_us=deadlines,median_mean_publication_after_embedding_dequeue_start_us=statistics.median(after)/1000,
        next_enqueue_has_not_started_at_end_of_embedding_consumer=sum(r["mean_enqueue_start_minus_embedding_dequeue_end_ns"]>0 for r in group))
with gzip.open(here/"requests.csv.gz","wt",encoding="utf8",newline="") as fp:
    out=csv.DictWriter(fp,fieldnames=list(full[0]));out.writeheader();out.writerows(full)
with trace.open("rb") as fp:
    digest=hashlib.file_digest(fp,"sha256").hexdigest()
(here/"results.json").write_text(json.dumps(dict(trace_sha256=digest,steps=outputs),indent=2))
