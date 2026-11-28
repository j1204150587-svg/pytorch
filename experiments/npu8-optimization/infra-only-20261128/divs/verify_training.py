"""Pair original training divisions by step, autograd worker and ATen ordinal; compare actual launches."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
from analyze_stock import summary, table


def collect(path):
    q=table(path/"queue_calls.csv.gz")
    tasks=table(path/"work_tasks.csv.gz")
    gaps=table(path/"device_gaps.csv.gz")
    owned=defaultdict(list)
    exposures=defaultdict(list)
    for t in tasks:
        owned[t["step"],t["queue_id"]].append(t)
    for g in gaps:
        exposures[g["step"],g["queue_id"]].append(g)
    pairs={}
    for c in q:
        if c["role"]!="worker" or c["aten"]!="aten::div" or c["op"] not in {"aclnnDivs","aclnnDiv"}:
            continue
        key=(int(c["step"]),int(c["aten_ordinal_in_thread"]))
        if key in pairs:
            raise ValueError(f"nonunique ATen worker ordinal {key}")
        pairs[key]=(c,owned[c["step"],c["queue_id"]],exposures[c["step"],c["queue_id"]])
    return pairs,tasks,q


def norm(tasks,step,ids):
    result=defaultdict(list)
    for t in tasks:
        if t["step"]!=str(step):
            continue
        if t["queue_id"] in ids:
            if "BroadcastTo" not in t["task"]:
                result[t["stream"]].append(("side_table_div",t["task_type"]))
        else:
            result[t["stream"]].append((t["task"],t["task_type"]))
    return dict(result)


def verify(a,b):
    raw=[collect(a),collect(b)]
    A,B=raw[0][0],raw[1][0]
    fields=("aten_to_enqueue_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","acl_api_sum_ns")
    gc=("before_aten_ns","aten_before_workspace_ns","workspace_ns","aten_to_enqueue_remainder_ns",
        "enqueue_ns","visible_queue_wait_ns","consumer_before_api_ns","acl_submission_api_ns","after_api_ns","gap_ns")
    rows=[]
    steps=[]
    for step in sorted({k[0] for k in A}):
        selected=[]
        for k,pa in sorted(A.items()):
            if k[0]!=step or k not in B:
                continue
            pb=B[k]
            if pa[0]["op"]=="aclnnDivs" and pb[0]["op"]=="aclnnDiv":
                selected.append((k,pa,pb))
        counts=[set(),set()]
        by_leg=[]
        report={"step":step,"eligible":len(selected)}
        if not selected:
            steps.append(report)
            continue
        report["shape"]=[int(selected[0][1][0][c]) for c in ("batch","seq_len")]
        scope_check=True
        for li,leg in enumerate("AB"):
            pcs=[s[li+1] for s in selected]
            qs=[c[0] for c in pcs]
            ww=[t for _,ts,_ in pcs for t in ts]
            gg=[g for _,_,gs in pcs for g in gs]
            counts[li]={q["queue_id"] for q in qs}
            expect=["aclnnDivs_BroadcastToAiCore_BroadcastTo","aclnnDivs_RealDivAiCore_RealDiv"] if li==0 else ["aclnnDiv_RealDivAiCore_RealDiv"]
            for (k,_,_), (q,ts,gs) in zip(selected,pcs):
                expected_stream=all(t["stream"]=="47" for t in ts)
                scope_check &= expected_stream and [t["task"] for t in ts]==expect and int(q["launch_api_count"])==len(expect)
                rows.append(dict(leg=leg,step=step,ordinal=k[1],queue_id=q["queue_id"],device_tasks="|".join(t["task"] for t in ts),
                                 task_ids="|".join(t["task_id"] for t in ts),streams="|".join(t["stream"] for t in ts),
                                 **{f:q[f] for f in fields},**{g+"__intersection_ns":sum(int(x[g] or 0) for x in gs) for g in gc}))
            report[leg]={"work":dict(Counter(t["task"] for t in ww)),
                         "actual_work":len(ww),"acl_launches":sum(int(q["launch_api_count"]) for q in qs),
                         "level1_all_work_by_stream":dict(Counter(t["stream"] for t in raw[li][1] if t["step"]==str(step))),
                         "latency":{f:summary([int(q[f])/1000 for q in qs]) for f in fields},
                         "gap_intersection_ms":{g:round(sum(int(x[g] or 0) for x in gg)/1e6,6) for g in gc}}
            by_leg.append(norm(raw[li][1], step,counts[li]))
        report["expected_targets_verified"]=bool(len(selected)==1024 and scope_check)
        report["normalized_streams_match"]=by_leg[0]==by_leg[1]
        report["normalized_stream_sha256"]=[{s:hashlib.sha256(json.dumps(v,separators=(",",":")).encode()).hexdigest()
                                             for s,v in x.items()} for x in by_leg]
        steps.append(report)
    return steps,rows


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("a",type=Path,help="Leg A outputs of analyze_dynamic_trace.py")
    ap.add_argument("b",type=Path,help="Leg B outputs of analyze_dynamic_trace.py")
    ap.add_argument("out",type=Path)
    args=ap.parse_args()
    steps,rows=verify(args.a,args.b)
    args.out.mkdir(exist_ok=True,parents=True)
    (args.out/"verified.json").write_text(json.dumps(steps,indent=2)+"\n",encoding="utf8")
    if rows:
        with gzip.open(args.out/"per_request.csv.gz","wt",newline="",encoding="utf8") as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
