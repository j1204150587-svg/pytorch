"""Recompute per-shape side-table latency and the associated all-stream gap durations."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import io
import json
import math
from pathlib import Path
import statistics
import tarfile


def rows(archive, name):
    with gzip.open(archive.extractfile(name), "rt", encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def latency(calls):
    output = {}
    for field in ("aten_to_enqueue_ns", "enqueue_ns", "signed_queue_wait_ns", "dequeue_ns", "acl_api_sum_ns"):
        values = sorted(int(row[field]) / 1000 for row in calls if row[field] != "")
        if values:
            output[field.removesuffix("_ns") + "_us"] = {
                "n": len(values), "p50": statistics.median(values),
                "p95": values[math.ceil(len(values) * .95) - 1],
                "p99": values[math.ceil(len(values) * .99) - 1], "max": values[-1],
            }
    return output


def read_leg(summary, raw, arm, leg):
    prefix = "level1/" + arm + "/linked/"
    calls = rows(summary, prefix + "queue_calls.csv.gz")
    tasks = defaultdict(list)
    for row in rows(summary, prefix + "work_tasks.csv.gz"):
        tasks[row["step"], row["queue_id"]].append(row)
    gaps = rows(summary, prefix + "device_gaps.csv.gz")
    kernel_csv, = (n for n in raw.getnames() if n.startswith("raw/") and "-l1-" in n
                   and f"/{leg}/" in n and n.endswith("/kernel_details.csv"))
    with io.TextIOWrapper(raw.extractfile(kernel_csv), encoding="utf-8-sig", newline="") as source:
        actual = defaultdict(list)
        for row in csv.DictReader(source):
            actual[row["Step Id"], row["Stream ID"], row["Task ID"], row["Name"]].append(row)
    result = {}
    for step in ("4", "5"):
        for op in ("aclnnEmbedding", "aclnnEmbeddingDenseBackward"):
            cohort = []
            kernels = Counter()
            for call in calls:
                if call["op"] != op or call["step"] != step:
                    continue
                work = tasks[step, call["queue_id"]]
                if not any("AsStrided" in t["task"] or "Slice" in t["task"] for t in work):
                    continue
                assert len(work) == int(call["launch_api_count"]) and len({t["stream"] for t in work}) == 1
                for t in work:
                    k, = actual[step, t["stream"], t["task_id"], t["task"]]
                    kernels[t["stream"], k["Name"], k["Input Shapes"], k["Input Data Types"]] += 1
                cohort.append(call)
            assert len(cohort) == 1024
            cohort.sort(key=lambda c: int(c["step_offset_ns"]))
            ids = {c["queue_id"] for c in cohort}
            assert len(ids) == 1024
            owned_gaps = [g for g in gaps if g["step"] == step and g["queue_id"] in ids and g["op"] == op]
            fields = ("gap_ns", "before_aten_ns", "workspace_ns", "aten_to_enqueue_remainder_ns",
                      "enqueue_ns", "visible_queue_wait_ns", "consumer_before_api_ns", "acl_submission_api_ns",
                      "after_api_ns")
            result.setdefault(step, {})[op] = {
                "shape": [int(cohort[0]["batch"]), int(cohort[0]["seq_len"])],
                "requests": len(cohort), "launches": sum(int(c["launch_api_count"]) for c in cohort),
                "first_table_us": latency(cohort[:1]), "subsequent_tables_us": latency(cohort[1:]),
                "first_table_workspace": int(bool(cohort[0]["workspace_ns"])),
                "subsequent_table_workspace": sum(bool(c["workspace_ns"]) for c in cohort[1:]),
                "nominal_gap_ms": {f: sum(int(g[f] or 0) for g in owned_gaps) / 1e6 for f in fields},
                "kernel_input_shapes": [dict(stream=k[0], name=k[1], shape=k[2], dtype=k[3], n=n)
                                        for k, n in kernels.most_common()],
            }
    return result


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory", type=Path)
    root = cli.parse_args().directory
    with tarfile.open(root / "summary.tgz") as summary, tarfile.open(root / "raw.tgz") as raw:
        report = {arm: read_leg(summary, raw, arm, leg) for arm, leg in (("A", "01-A"), ("B", "02-B"))}
    (root / "cohorts.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
