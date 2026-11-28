"""Identify real request cohorts and their gap overlap using the archived PTA/ACL timeline join."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys

def rows(p):
    with (gzip.open if p.suffix==".gz" else open)(p,"rt",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def summarize(values):
    s=sorted(values)
    if not s: return {"n":0}
    def quantile(p):
        rank=(len(s)-1)*p
        i=int(rank)
        return round(s[i]+(rank-i)*(s[min(i+1,len(s)-1)]-s[i]),3)
    return dict(n=len(s),p50=quantile(.5),p95=quantile(.95),p99=quantile(.99),max=quantile(1))

def cohort(parsed, leg, target):
    ops={"addmm":"aclnnAddmm","mm":"aclnnMm","emb-fwd":"aclnnEmbedding","emb-back":"aclnnEmbeddingDenseBackward"}
    calls=rows(parsed/"queue_calls.csv.gz")
    tasks = defaultdict(list)
    for r in rows(parsed/"work_tasks.csv.gz"):
        tasks[r["step"],r["queue_id"]].append(r)
    gaps={(r["step"],r["stream"],r["task_id"]):r for r in rows(parsed/"device_gaps.csv.gz")}
    kernel_root = Path("//?/" + leg.resolve().as_posix()) if os.name == "nt" else leg
    kernel_path,=kernel_root.rglob("kernel_details.csv")
    info={(r["Step Id"],r["Stream ID"],r["Task ID"]):r for r in rows(kernel_path)}
    result={}
    for step in ("4","5"):
        selected=[]
        for r in calls:
            if r["step"]!=step or r["op"]!=ops[target]: continue
            children=tasks[(step,r["queue_id"])]
            if target=="mm":
                # Restrict the 16 backward Matmul requests to the eight dInput calls with a static projection RHS.
                n=int(r["batch"])*int(r["seq_len"])
                shapes={info.get((step,k["stream"],k["task_id"]),{}).get("Input Shapes","").strip('"') for k in children}
                if not shapes & {f"{n},64;64,192",f"{n},256;256,64"}: continue
            selected.append((r,children))
        if not selected: raise ValueError(f"empty {target} at step {step}")
        raw=[r for r,_ in selected]
        work=[t for _,ts in selected for t in ts]
        gs=[gaps[(step,t["stream"],t["task_id"])] for t in work if (step,t["stream"],t["task_id"]) in gaps]
        timing=["aten_to_enqueue_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","acl_api_sum_ns","workspace_ns"]
        intervals=["before_aten_ns","workspace_ns","aten_to_enqueue_remainder_ns","enqueue_ns","visible_queue_wait_ns",
                   "consumer_before_api_ns","acl_submission_api_ns","after_api_ns","gap_ns"]
        signatures={r["role"]+":"+r["aten_ordinal_in_thread"]:[(t["stream"],t["stream_work_slot"],t["task_type"],t["task"]) for t in ts]
                    for r,ts in selected}
        result[step]=dict(requests=len(raw), launches=sum(int(r["launch_api_count"]) for r in raw),
            workspace=sum(bool(int(r["workspace_ns"] or 0)) for r in raw),
            work_by_stream=dict(Counter(t["stream"] for t in work)), kernels=dict(Counter(t["task"] for t in work)),
            latency_us={k:summarize([int(r[k])/1000 for r in raw if r.get(k)]) for k in timing},
            gap_ms={k:round(sum(int(g.get(k) or 0) for g in gs)/1e6,4) for k in intervals}, signatures=signatures)
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument("session",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("target",choices=("addmm","mm","emb-fwd","emb-back"))
    p.add_argument("--analyzers",type=Path,required=True)
    args=p.parse_args()
    legs={x.name[-1]:x for x in args.session.glob("0?-?")}
    if not (args.out/"submissions_compare.json").exists():
        subprocess.run([sys.executable,"-B",str(args.analyzers/"compare_dynamic_legs.py"),
                        str(legs["A"]),str(legs["B"]),str(args.out)],check=True)
    data={name:cohort(args.out/name,leg,args.target) for name,leg in legs.items()}
    matched={}
    for step in ("4","5"):
        a,b=(data[n][step].pop("signatures") for n in ("A","B"))
        keys=a.keys() & b.keys()
        matched[step]={"A_ordinals":len(a),"B_ordinals":len(b),"paired":len(keys),
            "same_device_sequence_and_stream":sum([(t[0],t[2],t[3]) for t in a[k]]==[(t[0],t[2],t[3]) for t in b[k]] for k in keys),
            "same_physical_slots_and_sequence":sum(a[k]==b[k] for k in keys)}
    (args.out/"cohort.json").write_text(json.dumps({"target":args.target,"cohorts":data,"matched":matched},indent=2))
if __name__=="__main__":
    main()

