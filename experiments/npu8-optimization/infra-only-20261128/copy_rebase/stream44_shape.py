"""Use the repeated native address signature to isolate actual 4096-BF16 slots on physical stream 44."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path


def load_csv(folder,name):
    with gzip.open(folder/(name+".csv.gz"),"rt",newline="") as source:return list(csv.DictReader(source))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("address",type=Path)
    p.add_argument("analysis",type=Path)
    cfg=p.parse_args()
    with cfg.address.open() as src:
        rows=[r for r in csv.DictReader(src,delimiter="\t") if r["physical_stream"]=="44"]
    target=list(range(25,1049))
    for offset in range(0,len(rows),1053):
        block=rows[offset:offset+1053]
        assert block[0]["src_start"]==block[0]["dst_start"]
        assert [i for i,r in enumerate(block[1:]) if r["dst_bytes"]=="8192"]==target
    out={}
    for leg in ("A","B"):
        root=cfg.analysis/leg
        calls={(r["step"],r["queue_id"]):r for r in load_csv(root,"queue_calls") if r["op"]=="aclnnInplaceCopy"}
        per_step=defaultdict(list)
        for row in load_csv(root,"work_tasks"):
            key=(row["step"],row["queue_id"])
            if row["stream"]=="44" and row["task"]=="MEMCPY_ASYNC" and key in calls:
                per_step[row["step"]].append((int(row["stream_work_slot"]),calls[key]))
        out[leg]={}
        for step,ordered in per_step.items():
            ordered.sort()
            assert len(ordered)==1052
            out[leg][step]={}
            for name,idx in (("target",target),("same_step_other",[i for i in range(1052) if i not in target]),
                             ("target_first_in_step",[target[0]]),("target_subsequent_in_step",target[1:])):
                selected=[ordered[i][1] for i in idx]
                metrics={}
                for f in ("aten_to_enqueue_ns","workspace_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","acl_api_sum_ns"):
                    vals=sorted(int(r[f]) for r in selected if r[f])
                    if vals:
                        metrics[f]=dict(n=len(vals),sum_ms=sum(vals)/1e6,**{s:vals[int((len(vals)-1)*fraction)]/1000
                            for s,fraction in (("p50_us",.5),("p95_us",.95),("p99_us",.99),("max_us",1))})
                out[leg][step][name]=metrics
    (cfg.analysis/"stream44_shapes.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
