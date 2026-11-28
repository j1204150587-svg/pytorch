"""Extract Level1 paired stage distributions for common high-frequency operations with matching device work."""
import argparse
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys


def read_csv(root, name):
    with gzip.open(root/(name+".csv.gz"),"rt",newline="") as src:return list(csv.DictReader(src))


def collect(root):
    work=defaultdict(list)
    for r in read_csv(root,"work_tasks"):
        work[r["step"],r["queue_id"]].append((r["stream"],r["stream_work_slot"],r["task"],r["task_type"]))
    gaps=defaultdict(list)
    for r in read_csv(root,"device_gaps"):
        if r["state"]=="linked":gaps[r["step"],r["queue_id"]].append(r)
    ops={"aclnnInplaceCopy","aclnnLinalgVectorNorm","aclnnDivs","aclnnAdd","aclnnMean","aclnnEmbedding","aclnnEmbeddingDenseBackward"}
    res={}
    for r in read_csv(root,"queue_calls"):
        st,qid=r["step"],r["queue_id"]
        key=(st,r["role"],r["aten_ordinal_in_thread"],r["aten"],r["op"])
        if r["op"] not in ops or not r["aten"] or not r["aten_ordinal_in_thread"]:continue
        assert key not in res
        res[key]=(r,work[st,qid],gaps[st,qid])
    return res


def summarize(items):
    vals=sorted(items)
    if not vals:return {"n":0}
    return dict(n=len(vals),sum_ms=sum(vals)/1e6,**{s:vals[int((len(vals)-1)*t)]/1000 for s,t in
        (("p50_us",.5),("p95_us",.95),("p99_us",.99),("max_us",1))})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("raw",type=Path)
    p.add_argument("out",type=Path)
    p.add_argument("tools",type=Path)
    args=p.parse_args()
    subprocess.run([sys.executable,"-B",str(args.tools/"compare_dynamic_legs.py"),str(args.raw/"01-A"),
        str(args.raw/"02-B"),str(args.out)],check=True)
    subprocess.run([sys.executable,"-B",str(args.tools/"stream_substitutions.py"),str(args.out)],check=True)
    a,b=collect(args.out/"A"),collect(args.out/"B")
    buckets=defaultdict(list)
    mismatch=Counter()
    for key in (a.keys()&b.keys()):
        ra,wa,ga=a[key]
        rb,wb,gb=b[key]
        if wa!=wb or ra["acl_api_sequence"]!=rb["acl_api_sequence"] or not wa:
            mismatch[key[0],ra["op"]]+=1
            continue
        buckets[key[0],ra["op"]].append((ra,ga,rb,gb))
    out={}
    for (st,op),rows in sorted(buckets.items()):
        summary={}
        for label,ri,gi in (("A",0,1),("B",2,3)):
            calls=[r[ri] for r in rows];gaps=[g for r in rows for g in r[gi]]
            summary[label]=dict(n=len(calls),gap_count=len(gaps),phases={col:summarize([int(r[col]) for r in calls if r[col]])
                for col in ("aten_to_enqueue_ns","workspace_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","acl_api_sum_ns")},
                gap_ms=sum(int(g["gap_ns"]) for g in gaps)/1e6,
                gap_intersections_ms={col:sum(int(g.get(col) or 0) for g in gaps)/1e6 for col in
                    ("before_aten_ns","workspace_ns","visible_queue_wait_ns","consumer_before_api_ns","acl_submission_api_ns","after_api_ns")})
        out[st+"/"+op]=dict(physically_matched=len(rows),unmatched=mismatch[st,op],**summary)
    (args.out/"paired_phases.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
