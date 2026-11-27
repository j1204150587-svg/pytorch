"""Measure HSTU core forward and sequence-matched backward CPU scopes in a Level0 leg."""
import argparse
from bisect import bisect_right
from collections import Counter,defaultdict
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path


def ns(x): return int(Decimal(str(x))*1000)


def merge(spans):
    merged=[]
    for a,b in sorted(spans):
        if a>=b:continue
        if merged and a<=merged[-1][1]:merged[-1]=(merged[-1][0],max(merged[-1][1],b))
        else:merged.append((a,b))
    return merged


def intersect(x,y):
    i=j=0; size=0
    while i<len(x) and j<len(y):
        a,b=x[i]; c,d=y[j]
        size+=max(0,min(b,d)-max(a,c))
        if b<=d:i+=1
        if d<=b:j+=1
    return size


def read_leg(leg,source_csv=None):
    profile=leg/"rank000/profile"
    trace=leg if leg.is_file() else leg/"trace_view.json.gz"
    if not trace.exists():
        files=list(profile.rglob("trace_view.json"))
        if len(files)!=1:raise ValueError(f"expected one trace in {leg}")
        trace=files[0]
    with (gzip.open if trace.suffix==".gz" else open)(trace,"rt",encoding="utf-8-sig") as f:
        doc=json.load(f)
    rows=doc["traceEvents"] if isinstance(doc,dict) else doc
    cpu=[]
    free=[]
    for row in rows:
        if row["ph"]!="X":continue
        a=ns(row["ts"]);b=a+ns(row["dur"]);name=row["name"]
        if name=="Free":free.append((a,b))
        if row.get("cat")=="cpu_op":
            cpu.append((a,b,name,row["tid"],row.get("args",{}).get("Sequence number",-1)))
    table=source_csv or leg/"step_trace_time.csv"
    if not table.is_file():table=next(profile.rglob("step_trace_time.csv"),None)
    steps={}
    if table:
        with table.open(encoding="utf-8-sig",newline="") as f:steps={int(x["Step"]):x for x in csv.DictReader(f)}
    return sorted(cpu),merge(free),steps


def report(leg,table=None):
    cpu,free,steps=read_leg(leg,table)
    output=[]
    for begin,end,marker,main_tid,_ in cpu:
        if not marker.startswith("ProfilerStep#"):continue
        k=int(marker.split("#")[1])
        rows=[r for r in cpu if begin<=r[0]<r[1]<=end]
        def scope(name):
            candidates=[r for r in rows if r[2]==name and r[3]==main_tid]
            if len(candidates)!=1:raise ValueError(f"expected one {name} in step {k}")
            return candidates[0][:2]
        fw=scope("hstu_forward_and_loss");bw=scope("deepspeed_backward")
        fw_cpu=[r for r in rows if r[3]==main_tid and fw[0]<=r[0] and r[1]<=fw[1]]
        core_start=min(r[0] for r in fw_cpu if r[2]=="aten::pow")
        core_end=min(r[0] for r in fw_cpu if r[2]=="aten::cross_entropy_loss")
        seq_sets=defaultdict(set)
        for a,b,op,tid,seq in fw_cpu:
            if seq>=0 and op.startswith("aten::"):
                seq_sets[seq].add("core" if core_start<=a<b<=core_end else "other")
        seq_core={s for s,labels in seq_sets.items() if labels=={"core"}}
        evals=[r for r in rows if r[2].startswith("autograd::engine::evaluate_function: ") and bw[0]<=r[0]<bw[1]]
        nodes=defaultdict(list)
        for a,b,op,tid,seq in rows:
            if seq>=0 and not op.startswith("aten::"):
                nodes[tid,op].append((a,b,seq))
        bwd_intervals=[]
        ops=Counter()
        counts=Counter()
        for a,b,label,tid,_ in evals:
            op=label.split(": ",1)[1]
            candidates=nodes[tid,op]
            pos=bisect_right(candidates,(a,-1,-1))
            if pos<len(candidates) and a<=candidates[pos][0]<candidates[pos][1]<=b and candidates[pos][2] in seq_core:
                bwd_intervals.append((a,b));ops[op]+=b-a;counts[op]+=1
        core=[(core_start,core_end)]
        bwd=merge(bwd_intervals)
        all_aten=merge([r[:2] for r in rows if r[2].startswith("aten::") and r[3]!=main_tid])
        fw_aten=merge([r[:2] for r in fw_cpu if r[2].startswith("aten::")])
        causal_names={"aten::ones","aten::tril","aten::bitwise_not","aten::masked_fill"}
        causal_fw=merge([r[:2] for r in fw_cpu if core_start<=r[0] and r[1]<=core_end and r[2] in causal_names])
        size=sum(b-a for a,b in bwd)
        base={}
        if k in steps:
            stage=float(steps[k]["Stage"])/1000; idle=float(steps[k]["Free"])/1000
            base.update(stage_ms=stage,free_ms=idle,nonfree_ms=stage-idle,
                        max_free_for_70_at_same_nonfree_ms=(stage-idle)*7/3,
                        min_cut_to_70_at_same_nonfree_ms=idle-(stage-idle)*7/3)
        output.append(dict(step=k,**base,core_fwd_ms=(core_end-core_start)/1e6,core_fwd_official_free_overlap_ms=intersect(core,free)/1e6,
            core_fwd_outside_aten_ms=(core_end-core_start-intersect(core,fw_aten))/1e6,
            core_bwd_evals=sum(counts.values()),core_bwd_eval_union_ms=size/1e6,
            core_bwd_eval_free_overlap_ms=intersect(bwd,free)/1e6,
            core_bwd_eval_outside_worker_aten_ms=(size-intersect(bwd,all_aten))/1e6,
            causal_fwd_marker_union_ms=sum(b-a for a,b in causal_fw)/1e6,
            causal_fwd_marker_free_overlap_ms=intersect(causal_fw,free)/1e6,
            core_backward_top=[dict(name=name,calls=counts[name],ms=value/1e6) for name,value in ops.most_common(10)]))
    return output


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg",type=Path);parser.add_argument("out",type=Path)
    parser.add_argument("--csv",type=Path)
    args=parser.parse_args()
    result=report(args.leg,args.csv)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps([{k:v for k,v in r.items() if k!="core_backward_top"} for r in result]))
