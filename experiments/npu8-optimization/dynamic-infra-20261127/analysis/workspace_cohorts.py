"""Measure same-leg requests split by their hardware tasks and visible workspace query."""
import argparse
import csv
from collections import defaultdict
import gzip
from pathlib import Path
from analyze_dynamic_trace import stats,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg",type=Path,help="analyzed leg containing work_tasks.csv.gz")
    parser.add_argument("op")
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    hardware=defaultdict(list)
    with gzip.open(args.leg/"work_tasks.csv.gz","rt",newline="") as f:
        for row in csv.DictReader(f):
            if row["queue_id"]:
                hardware[row["step"],row["queue_id"]].append(row["task"])
    groups=defaultdict(list)
    with gzip.open(args.leg/"queue_calls.csv.gz","rt",newline="") as f:
        for row in csv.DictReader(f):
            if row["op"]!=args.op:
                continue
            tasks=tuple(hardware[row["step"],row["queue_id"]])
            key=row["step"],tasks,bool(row.get("workspace_ns"))
            groups[key].append(row)
    out=[]
    for (step,tasks,queried),rows in sorted(groups.items()):
        for metric in ("aten_to_enqueue_ns","dequeue_ns","signed_queue_wait_ns","acl_api_sum_ns"):
            values=[int(row[metric]) for row in rows if row.get(metric)]
            out.append(dict(step=step,n_tasks=len(tasks),tasks="|".join(tasks),workspace_query_present=queried,
                            metric=metric,**stats(values)))
    write_csv(args.output,out)


if __name__=="__main__":
    main()
