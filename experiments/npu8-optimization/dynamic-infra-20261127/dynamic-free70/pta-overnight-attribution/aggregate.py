"""Summarize linked Level1 submissions and compare ordered physical work by stream."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path

TIME_FIELDS = ("aten_to_enqueue_ns", "enqueue_ns", "signed_queue_wait_ns", "dequeue_ns", "acl_api_sum_ns")
GAP_FIELDS = ("gap_ns", "before_aten_ns", "aten_before_workspace_ns", "workspace_ns",
              "aten_to_enqueue_remainder_ns", "enqueue_ns", "visible_queue_wait_ns",
              "consumer_before_api_ns", "acl_submission_api_ns", "after_api_ns")


def read_csv(root, name):
    with gzip.open(root / name, "rt", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def timings(rows, key):
    ordered = sorted(int(r[key])/1000 for r in rows if r[key])
    if not ordered:
        return {"n": 0}
    return {"n": len(ordered), **{f"p{pct}_us": round(ordered[(len(ordered)-1)*pct//100], 4)
                                 for pct in (50, 95, 99, 100)}}


def focus(data, gaps, works, step, target):
    qs = [r for r in data if int(r["step"]) == step and (target is None or r["op"] == target)]
    ids = {r["queue_id"] for r in qs}
    matched = [r for r in gaps if int(r["step"]) == step and r["queue_id"] in ids]
    work = [r for r in works if int(r["step"]) == step and r["queue_id"] in ids]
    return {"requests":len(qs),"joined_device_tasks":len(work),"linked_gaps":len(matched),
            "timing":{k:timings(qs,k) for k in TIME_FIELDS},
            "hardware_by_stream": dict(Counter(r["stream"] for r in work)),
            "overlap_ms":{k:round(sum(int(r[k] or 0) for r in matched)/1e6,6) for k in GAP_FIELDS}}


def summarize(root):
    legs = {}
    signatures = {}
    for label in ("01-A","02-B"):
        path = root / label
        report = json.loads((path/"submissions.json").read_bytes())
        queue = read_csv(path,"queue_calls.csv.gz")
        gaps = read_csv(path,"device_gaps.csv.gz")
        works = read_csv(path,"work_tasks.csv.gz")
        legs[label] = {"sha256":report["trace_sha256"], "steps":{}}
        signatures[label] = {}
        for sc in report["steps"]:
            step = sc["step"]
            signatures[label][step] = sc["streams"]
            legs[label]["steps"][step] = {"shape":[sc["batch"],sc["seq_len"]],
                "first_seen_step":sc["first_step_with_input_shape"],
                "all_linked":focus(queue,gaps,works,step,None),
                "embedding_backward":focus(queue,gaps,works,step,"aclnnEmbeddingDenseBackward")}
    comparisons = {}
    for step in signatures["01-A"]:
        a,b = [signatures[k][step] for k in ("01-A","02-B")]
        comparisons[step] = {k:{"same_names_types_in_order": a[k]["work_sha256"] == b[k]["work_sha256"],
                                "A":a[k]["work_n"],"B":b[k]["work_n"],
                                "all_with_controls_A":a[k]["all_n"],"all_with_controls_B":b[k]["all_n"]}
                             for k in sorted(a.keys() & b.keys())}
        if a.keys()!=b.keys():
            raise ValueError(f"{root}: physical stream set differs")
    return {"legs":legs,"physical_work_by_stream":comparisons,
            "quantiles":"floor(q*(n-1)), nonmissing linked records only",
            "gap_scope":"gaps before device work with a resolvable PTA request; CPU and device interval overlap"}


if __name__ == "__main__":
    cl = argparse.ArgumentParser(description=__doc__)
    cl.add_argument("root",type=Path)
    args=cl.parse_args()
    names = ("queue-short-poll-v1","repeat-embedding-stream-workspace-v2")
    data={n:summarize(args.root/n) for n in names}
    (args.root/"paired.json").write_text(json.dumps(data,indent=2)+"\n",encoding="utf8")
