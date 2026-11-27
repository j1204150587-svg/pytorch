"""Compare matched training steps, exact hardware work, the fold and four disjoint Host phases."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path
import sys

from fold_budget import analyze as fold_summary
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"causal_tril"))
from core_budget import intersect,merge,ns


def work(e):
    t=e.get("args",{}).get("Task Type","")
    return t in {"AI_CORE","AICORE","AI_VECTOR_CORE","VECTOR_CORE","AIC","AIV","MIX_AIC","MIX_AIV",
                 "MIX_AICORE","MIX_VECTOR_CORE","AI_CORE_MIX","MIX_AIC_AIV","MIX_AIV_AIC","AI_CPU","AICPU"} or \
                 t.startswith(("KERNEL_","SDMA","HDMA","RDMA","HCCS_DMA","PCIE_DMA","MEMCPY","MEMSET"))


def span(e):
    a=ns(e["ts"])
    return a,a+ns(e["dur"])


def digest(items):
    return hashlib.sha256(json.dumps(items,separators=(",",":"),ensure_ascii=False).encode()).hexdigest()


def leg_report(leg):
    path=leg/"trace_view.json.gz"
    folds={int(r["step"].split("#")[1]):r for r in fold_summary(path)}
    with gzip.open(path,"rt") as f:
        doc=json.load(f)
    events=doc if isinstance(doc,list) else doc["traceEvents"]
    pid,={r["pid"] for r in events if r["name"]=="process_name" and r["args"]["name"]=="Ascend Hardware"}
    hw=[r for r in events if r["ph"]=="X" and r["pid"]==pid and work(r)]
    eall=[r for r in events if r["ph"]=="X"]
    free=merge([span(r) for r in eall if r["name"]=="Free"])
    state=json.loads((leg/"rank000/eager_fold_plan.json").read_bytes())
    calls={r["step"]:r for r in state["steps"]}
    with (leg/"step_trace_time.csv").open(encoding="utf-8-sig",newline="") as f:
        official={int(r["Step"]):r for r in csv.DictReader(f)}
    result=[]
    for marker in (e for e in eall if e["name"].startswith("ProfilerStep#")):
        k=int(marker["name"].split("#")[1]);a,b=span(marker)
        inside=[r for r in eall if a<=span(r)[0]<b]
        cur=official[k]
        def one(name):
            row,=[r for r in inside if r["name"]==name]
            return row
        def phase(x,y):
            return {"wall_ms":(y-x)/1e6,"free_overlap_ms":intersect([(x,y)],free)/1e6}
        phases={name:phase(*span(one(name))) for name in ("input_generation_and_h2d","hstu_forward_and_loss",
                    "deepspeed_backward","optimizer_step_including_native_deepspeed_norm")}
        pstart,pend=span(one("hstu_forward_and_loss"));fstart,fend=span(one("OrderedTableSum"))
        inner={"before_fold":phase(pstart,fstart),"fold":phase(fstart,fend),"after_fold":phase(fend,pend)}
        tasks=[r for r in hw if a<=span(r)[0]<b]
        streams=defaultdict(list)
        for r in sorted(tasks,key=lambda e:(span(e)[0],e["args"].get("Task Id",-1),e["name"])):
            streams[r["args"]["Physic Stream Id"]].append((r["name"],r["args"]["Task Type"],r["args"].get("Task Id")))
        ftask,=[r for r in tasks if r["name"]=="ordered_64"]
        deq=one("Dequeue@ordered_64"); ds,de=span(deq)
        children=[(r["name"],float(r["dur"])) for r in eall if r is not deq and r["tid"]==deq["tid"]
                  and ds<=span(r)[0] and span(r)[1]<=de and r.get("cat")!="cpu_op"]
        endflows=[r for r in events if r["ph"]=="f" and r.get("cat")=="HostToDevice" and r["pid"]==ftask["pid"]
                   and r["tid"]==ftask["tid"] and ns(r["ts"])==span(ftask)[0]]
        origins=[]
        for flow in endflows:
            matching=[r for r in events if r["ph"]=="s" and r.get("cat")=="HostToDevice" and r["id"]==flow["id"]]
            destinations=sum(r["ph"]=="f" and r.get("cat")=="HostToDevice" and r["id"]==flow["id"] for r in events)
            origins.append({"id":flow["id"],"sources":len(matching),"destinations":destinations})
        result.append({"step":k,"receipt":calls[k],"free_ms":float(cur["Free"])/1000,"stage_ms":float(cur["Stage"])/1000,
            "computing_ms":float(cur["Computing"])/1000,"phases":phases,"forward_split":inner,
            "unaccounted_official_free_ms":float(cur["Free"])/1000-sum(x["free_overlap_ms"] for x in phases.values()),
            "fold":folds[k],"fold_task_id":ftask["args"].get("Task Id"),"fold_cann_children":children,"fold_host_to_device":origins,
            "step_work_tasks":len(tasks),"by_stream":{str(s):{"n":len(t),"names_types_hash":digest([v[:2] for v in t]),
                     "ids_names_types_hash":digest(t),"task_types":dict(Counter(v[1] for v in t))} for s,t in streams.items()}})
    return sorted(result,key=lambda r:r["step"])


def report(root):
    allruns={}
    for family in sorted(root.iterdir()):
        if not family.is_dir():continue
        ab={}
        for leg in sorted(family.iterdir()):
            if (leg/"trace_view.json.gz").is_file():
                ab[leg.name.split("-")[-1]]=leg_report(leg)
        if ab:
            allruns[family.name]=ab
    return allruns


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.write_text(json.dumps(report(args.evidence),indent=2)+"\n")
