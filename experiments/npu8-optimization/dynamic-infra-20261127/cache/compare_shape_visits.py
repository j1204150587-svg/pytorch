"""Pair fresh and revisited shape diagnostics by actual shape and verify work sequences."""
import argparse
import csv
import json
from pathlib import Path


def compare(first_dir, repeat_dir):
    indexes = [json.loads((d / "submissions.json").read_text())["steps"] for d in (first_dir, repeat_dir)]
    snapshots = []
    rows = []
    for first, repeat in zip(*indexes, strict=True):
        if (first["batch"], first["seq_len"], first["step"]) != (
                repeat["batch"], repeat["seq_len"], repeat["first_step_with_input_shape"]):
            raise ValueError("steps do not correspond to the first and second visit of the same shape")
        stream_match = (set(first["streams"]) == set(repeat["streams"]) and all(
            stream["work_n"] == repeat["streams"][idx]["work_n"] and
            stream["work_sha256"] == repeat["streams"][idx]["work_sha256"] for idx, stream in first["streams"].items()))
        snapshots.append(dict(shape=[first["batch"], first["seq_len"]], first=first["step"], repeat=repeat["step"],
                              same_ATen_thread_sequences=first["aten_thread_sequences"] == repeat["aten_thread_sequences"],
                              same_physical_stream_work=stream_match,
                              same_ACL_submissions=first["acl_submission_apis"] == repeat["acl_submission_apis"],
                              same_raw_hardware_types=first["hardware_types"] == repeat["hardware_types"]))
        for directory, step, visit in ((first_dir, first["step"], "first"), (repeat_dir, repeat["step"], "repeat")):
            with (directory / "operator_latency.csv").open(newline="") as stream:
                latency = [r for r in csv.DictReader(stream) if r["step"] == str(step)]
            for op in ("aclnnEmbedding", "aclnnEmbeddingDenseBackward"):
                matching = [r for r in latency if r["op"] == op]
                workspace_count = sum(int(r["n"]) for r in matching if r["metric"] == "workspace_ns")
                for row in matching:
                    rows.append(dict(visit=visit, step=step, batch=first["batch"], seq=first["seq_len"],
                                     op=op, workspace_count=workspace_count, metric=row["metric"],
                                     n=row["n"], p50_us=row["p50_us"], p95_us=row["p95_us"], p99_us=row["p99_us"]))
    (repeat_dir / "first-repeat-work.json").write_text(json.dumps(snapshots, indent=2) + "\n")
    with (repeat_dir / "first-repeat-latency.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fresh", type=Path)
    parser.add_argument("revisited", type=Path)
    args = parser.parse_args()
    compare(args.fresh, args.revisited)
