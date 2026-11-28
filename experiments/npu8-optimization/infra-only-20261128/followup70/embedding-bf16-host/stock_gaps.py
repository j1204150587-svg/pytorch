"""Measure all-stream idle intervals immediately preceding the two targeted stock Cast tasks."""
import argparse
import collections
import csv
import gzip
import hashlib
import json
import math
from decimal import Decimal
from pathlib import Path


def ns(row, field):
    return int(Decimal(row[field].strip()) * 1000)


def summary(values):
    s = sorted(values)
    return dict(calls=len(s), positive=sum(x>0 for x in s), total_ms=sum(s)/1e6,
                **{name:s[min(len(s)-1,math.ceil(q*len(s))-1)]/1000
                   for name,q in [("p50_us", .5),("p90_us", .9),("p99_us", .99),("max_us", 1.)]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input",type=Path)
    p.add_argument("coverage",type=Path)
    p.add_argument("output",type=Path)
    a = p.parse_args()
    with a.input.open(encoding="utf-8-sig",newline="") as f:
        rows=sorted(csv.DictReader(f),key=lambda r:ns(r,"Start Time(us)"))
    with gzip.open(a.coverage,"rt",encoding="utf-8-sig",newline="") as f:
        all_gaps=list(csv.DictReader(f))
    gaps={(r["step"],r["stream"],r["task_id"],r["task"]):r for r in all_gaps}
    assert len(gaps)==len(all_gaps)
    fields=("before_successor_both_outside_actual_submitters_ns","before_successor_other_submitting_aten_ns",
            "successor_aten_before_enqueue_ns","pre_enqueue_unknown_owner_ns","enqueue_in_progress_ns",
            "accepted_waiting_consumer_ns","dequeued_before_acl_ns","acl_in_progress_ns","acl_returned_ns","unlinked_ns")
    observations = collections.defaultdict(lambda: {"gaps":[],"duration":[],"stage":collections.Counter()})
    for row in rows:
        if row["Step Id"] not in ("4","5") or row["Type"]!="Cast" or not row["Name"].startswith("aclnnEmbeddingDenseBackward"):
            continue
        shape=row["Input Shapes"].strip('"')
        dtype=row["Input Data Types"]
        if dtype=="DT_BF16" and len(shape.split(","))==4:
            kind="grad_BF16_to_FP32"
        elif dtype=="FLOAT" and shape=="64,64":
            kind="result_FP32_to_BF16"
        else:
            continue
        stats=observations[(row["Step Id"],kind)]
        target=gaps.get((row["Step Id"],row["Stream ID"],row["Task ID"],row["Name"]))
        gap=int(target["gap_ns"]) if target else 0
        stage={k:int(target[k] or 0) for k in fields} if target else {}
        assert sum(stage.values())==gap
        stats["stage"].update(stage)
        stats["gaps"].append(gap)
        stats["duration"].append(ns(row,"Duration(us)"))
    report=dict(input_sha256=hashlib.sha256(a.input.read_bytes()).hexdigest(),kernel_rows=len(rows),
                coverage_sha256=hashlib.sha256(a.coverage.read_bytes()).hexdigest(),
                rule="Join all-device physical holes from producer-coverage to kernel CSV by (step,stream,task id,name). Nominal clock, no causal assertion.",
                groups=[dict(step=k[0],kind=k[1],right_adjacent_gap=summary(v["gaps"]),device_duration=summary(v["duration"]),
                             nominal_gap_stages_ms={name[:-3]:round(ns/1e6,6) for name,ns in v["stage"].items()})
                        for k,v in sorted(observations.items())])
    a.output.write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()

