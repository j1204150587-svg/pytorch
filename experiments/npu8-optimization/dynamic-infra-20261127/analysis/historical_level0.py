"""Recompute phase observations from the archived dynamic eager Level0 legs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def load(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()
    schedule = args.evidence / "dynamic_shapes_npu8.json"
    shapes = load(schedule)
    rows = []
    for leg in sorted(args.evidence.glob("*-l0-ab/01-A")):
        hashes = {name: digest for digest, name in
                  (line.split("  ", 1) for line in (leg.parent / "evidence.sha256").read_text().splitlines())}
        for path in ("command.json", "rank000/metrics.json", "trace_analysis.json", "step_trace_time.csv"):
            if hashlib.sha256((leg / path).read_bytes()).hexdigest() != hashes["01-A/" + path]:
                raise ValueError(f"hash mismatch in {leg / path}")
        plan = load(leg.parent / "plan.json")
        if hashlib.sha256(schedule.read_bytes()).hexdigest() != plan["shape_schedule_sha256"]:
            raise ValueError(f"schedule mismatch in {leg}")
        env = load(leg / "command.json")["env"]
        if env.get("TASK_QUEUE_ENABLE") != "1" or env.get("RECSYS_ZERO2_BATCH_OVERFLOW") != "1":
            raise ValueError(f"wrong queue or overflow strategy in {leg}")
        active = load(leg / "validated.json")
        if active["dynamic_norm"] or active["dynamic_partition"]:
            raise ValueError(f"unexpected training path in {leg}")
        metrics = {row["step"]: row for row in load(leg / "rank000/metrics.json")["steps"]}
        with (leg / "step_trace_time.csv").open(encoding="utf-8-sig", newline="") as handle:
            official = {int(row["Step"]): row for row in csv.DictReader(handle)}
        for step in load(leg / "trace_analysis.json")["steps"]:
            number = int(step["name"].split("#")[-1])
            batch, length = shapes[number]
            if (batch, length) != (metrics[number]["batch_size"], metrics[number]["seq_len"]):
                raise ValueError("recorded shape differs from schedule")
            for phase in [step] + step["stages"]:
                q = phase["same_host_clock_queue"]
                work, = phase["compute_aicpu_dma_in_host_window"].values()
                row = dict(run=leg.parent.name, step=number, batch=batch, seq_len=length,
                           first_step_with_input_shape=shapes.index([batch, length]),
                           phase=phase["name"], host_ms=phase["wall_ms"],
                           uncovered_work_ms=work["uncovered_ms"], matched_main_enqueue=q["paired_requests"],
                           adjacent_dequeue_gap_ms=q["adjacent_worker_gaps"].get("sum_ms", 0),
                           gap_before_subsequent_enqueue_ms=q["worker_gap_before_next_enqueue_start"].get("sum_ms", 0),
                           matched_dequeue_service_p50_us=q["dequeue_service"].get("median_ms", 0) * 1000,
                           matched_dequeue_service_p99_us=q["dequeue_service"].get("p99_ms", 0) * 1000)
                if phase is step:
                    for field in ("Free", "Stage", "Computing"):
                        row["official_" + field.lower() + "_ms"] = float(official[number][field]) / 1000
                rows.append(row)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
