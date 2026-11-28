"""Compare the same scalar Divs ordinals, kernel/stream slots and host intervals in the training pair."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

STUDY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(STUDY/"divs"))
from analyze_stock import table,summary

def collect(leg,output):
    if not (output/"submissions.json").is_file():
        analyzer=STUDY.parent/"dynamic-infra-20261127/analysis/analyze_dynamic_trace.py"
        subprocess.run([sys.executable,"-B",str(analyzer),str(leg),str(output)],check=True)
    path=Path(chr(92)*2+"?"+chr(92)+str(leg.resolve())) if sys.platform=="win32" else leg
    kernels=table(next(path.rglob("kernel_details.csv")))
    chosen={(k["Step Id"],k["Stream ID"],k["Task ID"]) for k in kernels if k["Type"]=="BroadcastTo" and ",1,64;" in k["Input Shapes"]}
    tasks=table(output/"work_tasks.csv.gz")
    ids={(t["step"],t["queue_id"]) for t in tasks if (t["step"],t["stream"],t["task_id"]) in chosen}
    calls={(q["step"],q["queue_id"]):q for q in table(output/"queue_calls.csv.gz") if q["op"]=="aclnnDivs" and q["aten"]=="aten::div" and q["role"]=="worker"}
    groups=defaultdict(list)
    for t in tasks:
        groups[t["step"],t["queue_id"]].append(t)
    gaps=defaultdict(list)
    for g in table(output/"device_gaps.csv.gz"):
        gaps[g["step"],g["queue_id"]].append(g)
    match={}
    for k in ids & calls.keys():
        q=calls[k]
        ordinal=int(q["aten_ordinal_in_thread"])
        wr=groups[k]
        key=(int(k[0]),ordinal)
        if key in match:
            raise ValueError(key)
        match[key]=(q,wr,gaps[k])
    sub=json.loads((output/"submissions.json").read_bytes())
    return match,{x["step"]:x["streams"] for x in sub["steps"]}


def run(a,b,out):
    legs=[collect(a,out/"A"),collect(b,out/"B")]
    if legs[0][0].keys()!=legs[1][0].keys():
        raise ValueError("Different side-table Divs ordinals")
    results=[]
    rows=[]
    fields=["aten_to_enqueue_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","acl_api_sum_ns"]
    gc=["before_aten_ns","aten_before_workspace_ns","workspace_ns","aten_to_enqueue_remainder_ns",
        "enqueue_ns","visible_queue_wait_ns","consumer_before_api_ns","acl_submission_api_ns","after_api_ns","gap_ns"]
    for step in sorted({k[0] for k in legs[0][0]}):
        selected=sorted(k for k in legs[0][0] if k[0]==step)
        s={"step":step,"pairs":len(selected)}
        align=True
        for k in selected:
            tasks=[[ (x["stream"],x["stream_work_slot"],x["task"],x["task_type"]) for x in leg[0][k][1] ] for leg in legs]
            align &= tasks[0]==tasks[1] and len(tasks[0])==2 and [t[2] for t in tasks[0]]==["aclnnDivs_BroadcastToAiCore_BroadcastTo","aclnnDivs_RealDivAiCore_RealDiv"]
        for name,(data,streams) in zip("AB",legs):
            selected_rows=[data[k] for k in selected]
            qs=[r[0] for r in selected_rows]
            gg=[g for _,_,gs in selected_rows for g in gs]
            s["shape"]=[int(qs[0]["batch"]),int(qs[0]["seq_len"])]
            s[name]={"work":dict(Counter(t["task"] for _,tt,_ in selected_rows for t in tt)),
                     "acl_launches":sum(int(q["launch_api_count"]) for q in qs),"workspace_calls":sum(bool(q["workspace_ns"]) for q in qs),
                     "physical_work_by_stream":{k:v["work_n"] for k,v in streams[step].items()},
                     "latency_us":{f:summary([int(q[f])/1000 for q in qs]) for f in fields},
                     "gap_intersection_ms":{c:round(sum(int(g[c] or 0) for g in gg)/1e6,6) for c in gc}}
            for k,(q,tt,gs) in zip(selected,selected_rows):
                rows.append(dict(leg=name,step=step,ordinal=k[1],queue_id=q["queue_id"],stream_slots="|".join(t["stream"]+":"+t["stream_work_slot"] for t in tt),
                                 **{f:q[f] for f in fields},**{c+"_exposed":sum(int(g[c] or 0) for g in gs) for c in gc}))
        fsets=[{k:(v["work_n"],v["work_sha256"]) for k,v in leg[1][step].items()} for leg in legs]
        s["same_whole_step_stream_work"]=fsets[0]==fsets[1]
        s["same_target_slots_and_work"]=bool(len(selected)==1024 and align and s["A"]["acl_launches"]==s["B"]["acl_launches"]==2048)
        results.append(s)
    (out/"verified.json").write_text(json.dumps(results,indent=2)+"\n")
    with gzip.open(out/"matched_requests.csv.gz","wt",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("a",type=Path)
    p.add_argument("b",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)
    run(args.a,args.b,args.out)
