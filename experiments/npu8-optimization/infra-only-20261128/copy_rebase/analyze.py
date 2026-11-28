"""Isolate actual one-SDMA InplaceCopy submissions by operator, step and physical stream."""

import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys


def read_csv(path):
    with gzip.open(path, "rt", newline="") as src:
        return list(csv.DictReader(src))


def collect(folder):
    calls = read_csv(folder / "queue_calls.csv.gz")
    tasks = defaultdict(list)
    for row in read_csv(folder / "work_tasks.csv.gz"):
        tasks[row["step"],row["queue_id"]].append(row)
    gaps = {(r["step"], r["stream"], r["task_id"]): r for r in read_csv(folder / "device_gaps.csv.gz")
            if r["state"] == "linked"}
    buckets = defaultdict(list)
    indexes = {}
    for call in calls:
        if call["op"] != "aclnnInplaceCopy" or call["aten"] != "aten::copy_":
            continue
        owned = tasks[call["step"], call["queue_id"]]
        if len(owned) != 1 or owned[0]["task"] != "MEMCPY_ASYNC" or not owned[0]["task_type"].startswith("SDMA"):
            continue
        task = owned[0]
        if call["acl_api_sequence"] != "AscendCL@aclrtMemcpyAsync":
            continue
        key = (call["step"], task["stream"], call["role"], call["aten_ordinal_in_thread"])
        if key in indexes:
            raise ValueError(f"ambiguous physical call ordinal {key}")
        indexes[key] = (int(task["stream_work_slot"]),task["task_type"],task["task"],call["acl_api_sequence"])
        gap = gaps.get((call["step"],task["stream"],task["task_id"]))
        buckets[call["step"],task["stream"]].append((call,gap))
    result = {}
    for (step, stream), pairs in sorted(buckets.items()):
        rows = [r for r,_ in pairs]
        gs = [g for _,g in pairs if g]
        measures = {}
        for k in ("aten_to_enqueue_ns", "enqueue_ns", "signed_queue_wait_ns", "dequeue_ns", "acl_api_sum_ns"):
            v = sorted(int(r[k])/1000 for r in rows if r[k])
            measures[k] = {"n":len(v),"p50_us":v[(len(v)-1)//2],"p95_us":v[int(.95*(len(v)-1))],
                           "p99_us":v[int(.99*(len(v)-1))],"max_us":max(v)}
        result[f"{step}/stream{stream}"] = dict(requests=len(rows), linked_gaps=len(gs),
            workspace_calls=sum(bool(r["workspace_ns"]) for r in rows),
            workspace_ms=sum(int(r["workspace_ns"] or 0) for r in rows)/1e6, latency=measures,
            gap_ms={k:sum(int(g[k] or 0) for g in gs)/1e6 for k in ("gap_ns","before_aten_ns","workspace_ns",
                    "visible_queue_wait_ns","consumer_before_api_ns","acl_submission_api_ns","after_api_ns")},
            onset=dict(Counter(g["first_visible"] for g in gs)))
    return indexes, result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("raw",type=Path)
    ap.add_argument("out",type=Path)
    ap.add_argument("--analyzers",type=Path,required=True)
    args=ap.parse_args()
    tools=args.analyzers
    def run(name,*params):
        subprocess.run([sys.executable,"-B",str(tools/name),*map(str,params)],check=True)
    run("compare_dynamic_legs.py",args.raw/"01-A",args.raw/"02-B",args.out)
    run("stream_substitutions.py",args.out)
    run("compare_operator_calls.py",args.out,"--aten","aten::copy_","--op","aclnnInplaceCopy","--out",args.out/"all-copy")
    ia,a=collect(args.out/"A")
    ib,b=collect(args.out/"B")
    pairs={}
    for step,stream in sorted({k[:2] for k in ia.keys()|ib.keys()}):
        ka={k for k in ia if k[:2]==(step,stream)}
        kb={k for k in ib if k[:2]==(step,stream)}
        pairs[f"{step}/stream{stream}"]=dict(A=len(ka),B=len(kb),both=len(ka&kb),
            identical_kind_slot_api=sum(ia[k]==ib[k] for k in ka&kb))
    (args.out/"single_sdma.json").write_text(json.dumps(dict(matched=pairs,A=a,B=b),indent=2)+"\n")


if __name__ == "__main__":
    main()
