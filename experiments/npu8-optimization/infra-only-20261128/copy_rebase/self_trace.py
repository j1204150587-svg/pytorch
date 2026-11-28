"""Find the queue request with zero device tasks preceding the stream-44 copy sequence."""
import argparse
from collections import defaultdict
import csv
import gzip
import json
from pathlib import Path


def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("calls",type=Path)
    cli.add_argument("trace",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    def rows(name):
        with gzip.open(args.calls/(name+".csv.gz"),"rt") as s:return list(csv.DictReader(s))
    work=defaultdict(list)
    for r in rows("work_tasks"):
        work[r["step"],r["queue_id"]].append(r)
    calls=rows("queue_calls")
    with gzip.open(args.trace,"rt",encoding="utf-8-sig") as s: trace=json.load(s)
    events=trace if isinstance(trace,list) else trace["traceEvents"]
    output={}
    for step in ("4","5"):
        selected=sorted((r for r in calls if r["step"]==step and r["op"]=="aclnnInplaceCopy" and r["role"]=="main"),
                        key=lambda r:int(r["step_offset_ns"]))
        i,=[i for i,r in enumerate(selected) if not work[step,r["queue_id"]]]
        candidate=selected[i]
        q,=[e for e in events if e.get("cat")=="enqueue" and str(e["args"]["correlation_id"])==candidate["queue_id"]]
        ts=float(q["ts"])
        op,=[e for e in events if e.get("ph")=="X" and e.get("name")=="aten::copy_" and e["tid"]==q["tid"] and
             float(e["ts"])<=ts<=float(e["ts"])+float(e["dur"])]
        group=selected[i+1:i+1053]
        assert len(group)==1052 and all([(t["stream"],t["task"]) for t in work[step,r["queue_id"]]]==
                                        [("44","MEMCPY_ASYNC")] for r in group)
        output[step]=dict(shape=[int(candidate["batch"]),int(candidate["seq_len"])],main_copy_index_1based=i+1,
            empty_queue_id=candidate["queue_id"],workspace_us=int(candidate["workspace_ns"])/1000,
            aten_us=float(op["dur"]),acl_sequence=candidate["acl_api_sequence"],
            acl_calls=int(candidate["launch_api_count"]),following_1052_queue_first=group[0]["queue_id"],
            following_1052_queue_last=group[-1]["queue_id"])
    args.output.write_text(json.dumps(output,indent=2)+"\n")


if __name__=="__main__":
    main()
