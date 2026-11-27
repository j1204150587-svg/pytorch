"""Count PTA requests, ACL calls and physical work on the IPG reduction stream."""
import argparse
import csv
import gzip
from collections import Counter
import json
from pathlib import Path


def table(p):
    with gzip.open(p,"rt",encoding="utf8") as f:
        return list(csv.DictReader(f))


def quantiles(us):
    vals=sorted(us)
    if not vals:return None
    return dict(n=len(vals),p50=vals[int((len(vals)-1)*.5)],p95=vals[int((len(vals)-1)*.95)],
                p99=vals[int((len(vals)-1)*.99)],maximum=vals[-1],sum=sum(vals))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("analysis",type=Path,help="output of analysis/analyze_dynamic_trace.py for one leg")
    cli.add_argument("out",type=Path)
    cli.add_argument("--stream",default="44")
    a=cli.parse_args()
    queue=table(a.analysis/"queue_calls.csv.gz")
    work=table(a.analysis/"work_tasks.csv.gz")
    gaps=table(a.analysis/"device_gaps.csv.gz")
    result=[]
    for step in sorted({r["step"] for r in queue},key=int):
        qs={r["queue_id"]:r for r in queue if r["step"]==step}
        ts=[r for r in work if r["step"]==step and r["stream"]==a.stream]
        selected=[r for r in ts if qs.get(r["queue_id"],{}).get("role")=="main"
                  and qs[r["queue_id"]]["aten"]=="aten::copy_"]
        ids={r["queue_id"] for r in selected}
        calls=[qs[q] for q in ids]
        delta=[r for r in gaps if r["step"]==step and r["queue_id"] in ids]
        physical=Counter((r["task_type"],r["task"]) for r in ts)
        result.append(dict(step=int(step),shape=[int(next(iter(qs.values()))[k]) for k in ("batch","seq_len")],
            stream=a.stream,all_tasks={str(k):v for k,v in physical.items()},
            main_copy_sdmas=sum(r["task_type"].startswith("SDMA") for r in selected),
            main_copy_tasks=len(selected),
            queue_requests=len(ids),acl_sequences=dict(Counter(q["acl_api_sequence"] for q in calls)),
            copy_preceding_gap_ms=sum(int(r["gap_ns"]) for r in delta)/1e6,
            timings_us={f:quantiles([int(q[f])/1e3 for q in calls if q.get(f) not in ("",None)])
                        for f in ("aten_to_enqueue_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","acl_api_sum_ns")}))
    a.out.write_text(json.dumps(result,indent=2)+"\n")
