"""Measure all-stream overlap for the Add after each side-table Mean, including gaps followed by other operators."""
import bisect
from collections import Counter,defaultdict
import csv
import gzip
import json
import math
from pathlib import Path
import statistics
import sys

STUDY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY/"div-geometry"))
from trace_budget import Span,Lanes,read


def run():
    d=STUDY/"embedding/analysis/emb-fwd/B"
    leg=STUDY/"evidence/raw/20261128T043752Z-ab-l1-7m6ais34/02-B"
    with gzip.open(leg/"trace_view.json.gz") as f:
        z=json.load(f)
    events=z if isinstance(z,list) else z["traceEvents"]
    names={"aten::add","empty_tensor","Enqueue@aclnnAdd","Dequeue@aclnnAdd","AscendCL@aclnnAddGetWorkspaceSize"}
    ss=[Span.read(e) for e in events if e.get("ph")=="X" and (e["name"] in names or e["name"].startswith(("ProfilerStep#","AscendCL@aclrtLaunchKernel")))]
    owner=Lanes([s for s in ss if s.name=="aten::add"])
    empty=Lanes([s for s in ss if s.name=="empty_tensor"])
    acl=Lanes([s for s in ss if s.name.startswith("AscendCL@aclrtLaunchKernel")])
    ws=Lanes([s for s in ss if s.name=="AscendCL@aclnnAddGetWorkspaceSize"])
    enq={str(s.args["correlation_id"]):s for s in ss if s.name=="Enqueue@aclnnAdd"}
    deq={str(s.args["correlation_id"]):s for s in ss if s.name=="Dequeue@aclnnAdd"}
    steps={s.name.split("#")[-1]:s for s in ss if s.name.startswith("ProfilerStep#")}
    work=read(d/"work_tasks.csv.gz")
    side={(t["step"],t["queue_id"]) for t in work if t["task"]=="aclnnEmbedding_AsStridedAiCore_AsStrided"}
    hardware=defaultdict(list)
    for t in work:
        hardware[t["step"],t["queue_id"]].append(t)
    queue=read(d/"queue_calls.csv.gz")
    gs=read(d/"device_gaps.csv.gz")
    report={}
    raw=[]
    for step,sc in steps.items():
        calls=sorted((q for q in queue if q["step"]==step and q["role"]=="main"),key=lambda x:int(x["step_offset_ns"]))
        selected=[]
        for i,q in enumerate(calls[:-2]):
            if (step,q["queue_id"]) in side:
                if calls[i+1]["op"]!="aclnnMean":
                    raise ValueError((step,i,calls[i+1]["op"]))
                if calls[i+2]["op"]=="aclnnAdd":
                    selected.append(calls[i+2])
        gaps=sorted((sc.start+int(g["step_offset_ns"]),int(g["gap_ns"]),g["op"]) for g in gs if g["step"]==step)
        starts=[g[0] for g in gaps]
        hold=defaultdict(Counter)
        def stat(tag,l,r):
            if r<l:
                raise ValueError((tag,l,r))
            hold[tag]["cpu_ns"]+=r-l
            i=max(0,bisect.bisect_right(starts,l)-1)
            while i<len(gaps) and gaps[i][0]<r:
                t,d,op=gaps[i]
                ov=max(0,min(r,t+d)-max(l,t))
                hold[tag]["overlap_ns"]+=ov
                if op=="aclnnAdd":
                    hold[tag]["overlap_before_any_add_ns"]+=ov
                i+=1
        names=Counter()
        workspaces=0
        own_gaps=defaultdict(int)
        for g in gs:
            if g["step"]==step:
                own_gaps[g["queue_id"]]+=int(g["gap_ns"])
        for ordinal,q in enumerate(selected,1):
            cid=q["queue_id"]
            en,de=enq[cid],deq[cid]
            host=owner.parent(en)
            for s in ws.children(host):
                stat("workspace",s.start,s.end)
                workspaces+=1
            e=[t for t in empty.children(host) if t.end<=en.start]
            if len(e)!=1:
                raise ValueError(("output allocations",len(e)))
            stat("producer_pre_enqueue_inclusive",host.start,en.start)
            stat("output_empty",e[0].start,e[0].end)
            stat("enqueue",en.start,en.end)
            launch=acl.children(de)
            if len(launch)!=1:
                raise ValueError(("CANN submits",len(launch)))
            raw.append(dict(step=int(step),batch=int(q["batch"]),seq_len=int(q["seq_len"]),side_add_ordinal=ordinal,
                            queue_id=cid,workspace_query_count=len(ws.children(host)),producer_ns=en.start-host.start,
                            output_allocation_ns=e[0].end-e[0].start,enqueue_ns=en.end-en.start,
                            signed_queue_wait_ns=de.start-en.end,consumer_ns=de.end-de.start,
                            consumer_pre_acl_ns=launch[0].start-de.start,acl_ns=launch[0].end-launch[0].start,
                            attached_all_stream_gap_ns=own_gaps[cid]))
            stat("consumer_pre_acl",de.start,launch[0].start)
            stat("acl",launch[0].start,launch[0].end)
            stat("consumer_post_acl",launch[0].end,de.end)
            for t in hardware[step,cid]:
                names[t["stream"]+" "+t["task"]]+=1
        def conv(c):
            return {k.replace("_ns","_ms"):round(v/1e6,6) for k,v in c.items()}
        report[step]=dict(calls=len(selected),workspace_calls=workspaces,device_work=names,
            intervals={k:conv(v) for k,v in hold.items()},
            producer_outside_allocation_exposed_ms=round((hold["producer_pre_enqueue_inclusive"]["overlap_ns"]-hold["output_empty"]["overlap_ns"])/1e6,6))
        rows=[r for r in raw if r["step"]==int(step)]
        tails={}
        for k in [k for k in rows[0] if k.endswith("_ns")]:
            vals=sorted(r[k]/1000 for r in rows[1:])
            worst=max(rows,key=lambda r:r[k])
            tails[k[:-3]+"_us"]=dict(first=round(rows[0][k]/1000,3),subsequent_median=round(statistics.median(vals),3),
                subsequent_p95=vals[math.ceil(.95*len(vals))-1],subsequent_p99=vals[math.ceil(.99*len(vals))-1],
                subsequent_max=max(vals),max_ordinal=worst["side_add_ordinal"],max_row_attached_gap_us=worst["attached_all_stream_gap_ns"]/1000)
        report[step]["within_step"]=tails
        report[step]["workspace_query_ordinals"]=[r["side_add_ordinal"] for r in rows if r["workspace_query_count"]]
    with gzip.open(Path(__file__).with_name("side-add-requests.csv.gz"),"wt",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=raw[0])
        writer.writeheader()
        writer.writerows(raw)
    return report

if __name__=="__main__":
    Path(__file__).with_name("admission.json").write_text(json.dumps(run(),indent=2)+"\n")
