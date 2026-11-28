"""Audit real training strided-index tasks and the isolated ACL 2-D copy probe."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
import tarfile
from decimal import Decimal
from pathlib import Path


def latency(us):
    s=sorted(us)
    return dict(n=len(s),sum_ms=sum(s)/1000,p50_us=s[math.ceil(len(s)*.50)-1],p99_us=s[math.ceil(len(s)*.99)-1],max_us=s[-1])


def training(kernel,coverage):
    with gzip.open(coverage,"rt",newline="") as f:
        gaps={(r["step"],r["stream"],r["task_id"],r["task"]):r for r in csv.DictReader(f)}
    result=collections.defaultdict(list)
    with kernel.open(encoding="utf-8-sig",newline="") as f:
        for r in csv.DictReader(f):
            if r["Type"]=="AsStrided" and r["Name"].startswith("aclnnEmbedding") and r["Step Id"] in ("4","5"):
                key=(r["Step Id"],r["Stream ID"],r["Task ID"],r["Name"])
                g=gaps.get(key,{})
                result[(r["Step Id"],r["Name"],r["Stream ID"])].append(
                    (float(r["Duration(us)"]),int(g.get("gap_ns",0))/1000,g))
    stages=("before_successor_both_outside_actual_submitters_ns","before_successor_other_submitting_aten_ns",
            "successor_aten_before_enqueue_ns","pre_enqueue_unknown_owner_ns","enqueue_in_progress_ns",
            "accepted_waiting_consumer_ns","dequeued_before_acl_ns","acl_in_progress_ns","acl_returned_ns","unlinked_ns")
    groups=[]
    for k,v in sorted(result.items()):
        sums={s[:-3]:sum(int(r.get(s,0) or 0) for _,__,r in v)/1e6 for s in stages}
        assert abs(sum(sums.values())-sum(g for _,g,__ in v)/1e3)<1e-7
        groups.append(dict(step=k[0],op=k[1],stream=k[2],actual_tasks=len(v),
                           device_work=latency([a for a,_,__ in v]),right_gap=latency([b for _,b,__ in v]),
                           nominal_stages_ms=sums))
    return dict(kernel_sha256=hashlib.sha256(kernel.read_bytes()).hexdigest(),
                coverage_sha256=hashlib.sha256(coverage.read_bytes()).hexdigest(),groups=groups)


def actual(arc):
    with tarfile.open(arc,"r:gz") as tar:
        rows=json.load(tar.extractfile("run/results.json"))
        assert tar.extractfile("run/train.exit").read().strip()==b"0" and rows["complete"]
        trace_name,=[n for n in tar.getnames() if n.endswith("/trace_view.json")]
        tr=json.load(tar.extractfile(trace_name),parse_float=Decimal)
    tr=tr if isinstance(tr,list) else tr["traceEvents"]
    hw,=[e["pid"] for e in tr if e["name"]=="process_name" and e.get("args",{}).get("name")=="Ascend Hardware"]
    cann,=[e["pid"] for e in tr if e["name"]=="process_name" and e.get("args",{}).get("name")=="CANN"]
    markers={e["name"]:e for e in tr if e.get("ph")=="X" and e["name"].startswith("studyDMA_")}
    packets=[]
    for r in rows["profiles"]:
        m=markers[r["label"]]
        lo,hi=Decimal(m["ts"]),Decimal(m["ts"])+Decimal(m["dur"])
        relevant=[e for e in tr if e.get("ph")=="X" and lo<=Decimal(e["ts"])<hi]
        device=[dict(name=e["name"],task=e["args"].get("Task Id"),stream=e["args"].get("Physic Stream Id"),
                     kind=e["args"].get("Task Type"),duration_us=float(e["dur"])) for e in relevant if e["pid"]==hw]
        apis=[e["name"] for e in relevant if e["pid"]==cann and e["name"].startswith("AscendCL@")]
        packets.append(dict(label=r["label"],shape=r["shape"],table=r["table"],rc=r["rc"],
                            kernels=device,acl_apis=dict(collections.Counter(apis))))
    groups={}
    for leg in ("ref","D2D1","D2D2"):
        batch=[x for x in rows["results"] if x["leg"]==leg]
        assert len(batch)==60
        groups[leg]=dict(rc=dict(collections.Counter(r["rc"] for r in batch)),sync_rc=dict(collections.Counter(r["sync_rc"] for r in batch)),
                         exact=sum(r["ref_word_differences"]==0 for r in batch),all_sentinel=sum(r["all_sentinel"] for r in batch),
                         host_call=latency([r["host_call_ns"]/1000 for r in batch]),sync=latency([r["sync_ns"]/1000 for r in batch]))
    return dict(soc=rows["soc"],unprofiled=groups,device=packets)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind",choices=("training","probe"))
    p.add_argument("inputs",nargs="+",type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    data=training(*args.inputs) if args.kind=="training" else actual(*args.inputs)
    args.out.write_text(json.dumps(data,indent=2),encoding="utf-8")

