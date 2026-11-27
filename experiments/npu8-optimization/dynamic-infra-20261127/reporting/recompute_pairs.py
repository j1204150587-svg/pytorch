"""Recompute dynamic HSTU paired synchronized timings and profiler windows from raw receipts."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


COLUMNS = ["优化前端到端", "优化后端到端", "端到端提升比例", "优化前 Free", "优化前 Free 占比",
           "优化后 Free", "优化后 Free 占比", "Free 占比提升比例"]
PROFILE_FIELDS = {"free_ms": "Free", "stage_ms": "Stage", "computing_ms": "Computing",
                  "communication_not_overlapped_ms": "Communication(Not Overlapped)"}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def raw_file(leg, rank, filename):
    for parent in (leg, rank):
        direct = parent / filename
        if direct.is_file():
            return direct
    paths = list(rank.rglob(filename))
    check(len(paths) == 1, f"expected exactly one {filename} under {rank}")
    return paths[0]


def profile(leg, rank, shapes, mode):
    path = raw_file(leg, rank, "step_trace_time.csv")
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    ids = [int(row["Step"]) for row in rows]
    expected_ids = {"l0": [4, 5, 6], "l1": [4, 5], "l1-late": [12, 13]}[mode]
    check(ids == expected_ids, f"{path}: unexpected profile steps {ids}")
    check(len({row["Device_id"] for row in rows}) == 1, f"{path}: mixed devices")
    analysis_path = leg / "trace_analysis.json"
    host_windows = {}
    if analysis_path.is_file():
        host_windows = {r["name"]: r["wall_ms"] for r in read(analysis_path)["steps"]}
        check(set(host_windows) == {f"ProfilerStep#{n}" for n in expected_ids},
              f"{leg}: CPU step markers differ from the CSV")
    per_step = []
    for step, row in zip(ids, rows):
        entry = {name: float(row[column]) / 1000 for name, column in PROFILE_FIELDS.items()}
        check(all(math.isfinite(v) and v >= 0 for v in entry.values()) and entry["stage_ms"] > 0,
              f"{path}: invalid profile durations at {step}")
        check(math.isclose(entry["stage_ms"], entry["free_ms"] + entry["computing_ms"] +
                           entry["communication_not_overlapped_ms"], rel_tol=0, abs_tol=0.02),
              f"{path}: Stage identity failed at {step}")
        record = {"step": step, "batch_size": shapes[step][0], "seq_len": shapes[step][1],
                  "device_id": row["Device_id"], **entry}
        if host_windows:
            wall = host_windows[f"ProfilerStep#{step}"]
            record.update(cpu_marker_ms=wall, cpu_marker_minus_stage_ms=wall - entry["stage_ms"])
        per_step.append(record)
    means = {name: statistics.fmean(row[name] for row in per_step) for name in PROFILE_FIELDS}
    result = {"csv": str(path), "per_step": per_step, **means, "free_fraction": means["free_ms"] / means["stage_ms"]}
    if mode == "l0":
        result["first_two_steps_free_fraction_diagnostic"] = (sum(r["free_ms"] for r in per_step[:2]) /
                                                              sum(r["stage_ms"] for r in per_step[:2]))
    return result


def steady(summary, shapes, rank):
    window = summary["steady_state"]
    check((window["start_step_inclusive"], window["end_step_exclusive"], window["steps"]) == (10, 60, 50),
          f"{rank}: expected synchronized [10,60) window")
    actual = shapes[10:60]
    samples = sum(b for b, _ in actual)
    tokens = sum(b * n for b, n in actual)
    check((samples, tokens) == (window["samples"], window["tokens"]), f"{rank}: window work mismatch")
    elapsed = float(window["elapsed_s_synchronized"])
    check(math.isfinite(elapsed) and elapsed > 0, f"{rank}: invalid elapsed time")
    rates = {"ms_per_step": elapsed * 1000 / 50, "samples_per_s": samples / elapsed, "tokens_per_s": tokens / elapsed}
    check(all(math.isclose(window[k], v, rel_tol=1e-8, abs_tol=1e-8) for k, v in rates.items()),
          f"{rank}: reported window rates disagree")
    return {**rates, "elapsed_s": elapsed, "window_samples": samples, "window_tokens": tokens, "window_steps": 50}


def parse_leg(path, variant, mode, schedule):
    rank = next((p for p in (path / "rank000", path / "training/rank000", path)
                 if (p / "metrics.json").is_file()), None)
    check(rank is not None, f"{path}: missing raw rank000/metrics.json")
    metrics = read(rank / "metrics.json")
    rows, summary = metrics["steps"], metrics["summary"]
    shapes = [[r["batch_size"], r["seq_len"]] for r in rows]
    count = {"no-profiler": 60, "l0": 8, "l1": 8, "l1-late": 16}[mode]
    check(len(rows) == count and [r["step"] for r in rows] == list(range(count)) and shapes == schedule[:count],
          f"{path}: steps or shapes differ from the supplied workload")
    check(summary["status"] == "completed" and summary["steps"] == count and
          summary["profile_enabled"] == (mode != "no-profiler"), f"{path}: wrong training status or mode")
    command_file = path / "command.json"
    command = read(command_file) if command_file.is_file() else {}
    if command.get("variant") is not None:
        check(command["variant"] == variant, f"{path}: variant mismatch")
    env = command.get("env")
    if env is not None:
        check(env.get("TASK_QUEUE_ENABLE") == "1" and env.get("RECSYS_ZERO2_BATCH_OVERFLOW") == "1",
              f"{path}: expected 01 dynamic eager TQ1")
    args_file = rank / "run_arguments.json"
    args = read(args_file) if args_file.is_file() else {}
    if args:
        check(args["steps"] == count and args["seed"] == 1234 and args["shape_schedule"] and
              args["profile_level1"] == mode.startswith("l1") and args["profile"] == (mode != "no-profiler"),
              f"{path}: unexpected training arguments")
    config_path = rank / "effective_deepspeed_config.json"
    config = read(config_path) if config_path.is_file() else {}
    if config:
        check(config["bf16"]["enabled"] and config["bf16"]["check_grad_overflow"] and
              config["zero_optimization"]["stage"] == 2 and config["gradient_accumulation_steps"] == 1,
              f"{path}: unexpected DeepSpeed settings")
    check(not (rank / "hstu_graph.json").exists(), f"{rank}: unexpected HSTU graph receipt")
    activation = rank / "overflow_scan.json"
    scan = read(activation) if activation.is_file() else {}
    if scan:
        check(scan["valid"] and scan["calls"] == scan["fast_calls"] == count and
              scan["fallback_calls"] == scan["errors"] == 0, f"{rank}: unexpected 01 activations")
    result = (steady(summary, shapes, rank) if mode == "no-profiler" else profile(path, rank, shapes, mode))
    if mode == "no-profiler":
        result["host_step_ms"] = [float(r["host_step_ms"]) for r in rows[10:60]]
    return {"variant": variant, "directory": str(path), "metrics": str(rank / "metrics.json"),
            "environment": env, "arguments": {k: v for k, v in args.items() if k != "output_dir"},
            "deepspeed": config, "shape_count": len(set(map(tuple, shapes))), "overflow_scan": scan,
            "sampled_loss": {r["step"]: r["loss"] for r in rows if "loss" in r}, **result}


def parse_pair(spec, schedule, switch):
    name, directory = spec.split("=", 1)
    series, order, mode = name.split("/")
    check(series and order in ("ab", "ba", "a", "b", "aa") and mode in ("l0", "l1", "l1-late", "no-profiler"),
          f"invalid --pair expression: {spec}")
    path = Path(directory)
    if (path / "status.json").is_file():
        status = read(path / "status.json")
        if "exit" in status:
            check(status["exit"] == 0, f"{path}: failed session")
    subdirs = sorted((p for p in path.iterdir() if p.is_dir() and re.fullmatch(r"\d+-[AB]", p.name)),
                     key=lambda p: int(p.name.split("-")[0]))
    if len(order) == 1 and not subdirs:
        subdirs = [path]
    else:
        check("".join(p.name[-1] for p in subdirs).lower() == order, f"{path}: leg order disagrees with {order}")
    legs = [parse_leg(p, v, mode, schedule) for v, p in zip(order.upper(), subdirs, strict=True)]
    pair = {"series": series, "order": order.upper(), "mode": mode, "directory": str(path), "legs": legs}
    if len(legs) == 1:
        return pair
    a, b = (legs if order != "ba" else reversed(legs))
    for field in ("arguments", "deepspeed"):
        check(a[field] == b[field], f"{path}: A/B {field} differ")
    if a["environment"] is not None and b["environment"] is not None:
        env_a, env_b = a["environment"], b["environment"]
        changes = {k: [env_a.get(k), env_b.get(k)] for k in env_a.keys() | env_b.keys() if env_a.get(k) != env_b.get(k)}
        pair["changed_environment"] = changes
        if switch:
            check(set(changes) == (set() if order == "aa" else {switch}), f"{path}: unexpected env changes {changes}")
    losses_a, losses_b = a["sampled_loss"], b["sampled_loss"]
    check(losses_a.keys() == losses_b.keys(), f"{path}: mismatched logged steps")
    pair["sampled_loss_max_difference"] = max((abs(x - losses_b[i]) for i, x in losses_a.items()), default=None)
    pair["sampled_losses_close_1e_6"] = all(math.isclose(x, losses_b[i], rel_tol=1e-6, abs_tol=1e-6)
                                             for i, x in losses_a.items())
    if mode == "no-profiler":
        pair.update(saved_ms_per_step=a["ms_per_step"] - b["ms_per_step"],
                    step_gain=1 - b["ms_per_step"] / a["ms_per_step"])
        # Only the entire 50-step window is synchronized. Each training-loop wall time can include implicit waits.
        host = [{"step": i, "batch_size": schedule[i][0], "seq_len": schedule[i][1],
                 "host_a_ms": x, "host_b_ms": y, "host_a_minus_b_ms": x-y}
                for i, (x, y) in enumerate(zip(a["host_step_ms"], b["host_step_ms"], strict=True), 10)]
        deltas = sorted(r["host_a_minus_b_ms"] for r in host)
        pair.update(host_step_pairs=host, training_loop_wall_delta_ms={"min": deltas[0],
                    "p10": deltas[int(0.1 * (len(deltas)-1))], "median": statistics.median(deltas),
                    "p90": deltas[int(0.9 * (len(deltas)-1))], "max": deltas[-1]})
    else:
        pair.update(saved_free_ms=a["free_ms"] - b["free_ms"],
                    free_fraction_gain=1 - b["free_fraction"] / a["free_fraction"] if a["free_fraction"] else None)
    return pair


def summarize(pairs):
    results = {}
    for p in pairs:
        key = (p["series"], p["order"])
        check(p["mode"] not in results.setdefault(key, {}), f"duplicate mode for {key}, use a distinct series")
        results[key][p["mode"]] = p
    rows = []
    for (series, order), modes in results.items():
        if not {"no-profiler", "l0"} <= modes.keys() or order not in ("AB", "BA"):
            continue
        e2e, free = modes["no-profiler"], modes["l0"]
        e, f = [{v["variant"]: v for v in p["legs"]} for p in (e2e, free)]
        values = [e["A"]["ms_per_step"], e["B"]["ms_per_step"], e2e["step_gain"],
                  f["A"]["free_ms"], f["A"]["free_fraction"], f["B"]["free_ms"], f["B"]["free_fraction"],
                  free["free_fraction_gain"]]
        rows.append({"series": series, "order": order, "cells": values})
    paired = [p for p in pairs if p["mode"] == "no-profiler" and p["order"] in ("AB", "BA")]
    variation = {}
    for name in ("step_gain", "saved_ms_per_step"):
        measures = [p[name] for p in paired]
        if measures:
            variation[name] = {"n": len(measures), "min": min(measures), "max": max(measures),
                               "range": max(measures)-min(measures), "median": statistics.median(measures)}
    return rows, variation


def markdown(rows):
    table = []
    for row in rows:
        table.extend([f'**{row["series"]} · {row["order"]}**', "",
                      "| " + " | ".join(COLUMNS) + " |", "|" + "---:|" * 8])
        values = []
        for i, v in enumerate(row["cells"]):
            if v is None:
                values.append("—")
            elif i in (0, 1, 3, 5):
                values.append(f"{v:.3f} ms/步")
            elif i in (2, 7):
                values.append(f"{v:+.2%}")
            else:
                values.append(f"{v:.2%}")
        table.extend(["| " + " | ".join(values) + " |", ""])
    return "\n".join(table)


def write_tables(out, pairs):
    steady_columns = ["series", "order", "variant", "window_steps", "window_samples", "window_tokens", "elapsed_s",
                      "ms_per_step", "samples_per_s", "tokens_per_s", "saved_ms_per_step", "step_gain"]
    step_columns = ["series", "order", "mode", "variant", "step", "batch_size", "seq_len", "device_id",
                    *PROFILE_FIELDS.keys(), "cpu_marker_ms", "cpu_marker_minus_stage_ms"]
    with (out / "synchronized_windows.csv").open("w", newline="", encoding="utf-8") as windows, (
         out / "profile_steps.csv").open("w", newline="", encoding="utf-8") as steps:
        w = csv.DictWriter(windows, fieldnames=steady_columns, extrasaction="ignore")
        s = csv.DictWriter(steps, fieldnames=step_columns, extrasaction="ignore")
        w.writeheader()
        s.writeheader()
        for pair in pairs:
            for leg in pair["legs"]:
                keys = {"series": pair["series"], "order": pair["order"], "mode": pair["mode"],
                        "variant": leg["variant"]}
                if pair["mode"] == "no-profiler":
                    w.writerow({**leg, **keys, "saved_ms_per_step": pair.get("saved_ms_per_step"),
                                "step_gain": pair.get("step_gain")})
                else:
                    for row in leg["per_step"]:
                        s.writerow({**row, **keys})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--switch", help="enforce a single changed environment variable for an A/B pair")
    parser.add_argument("--pair", action="append", required=True, metavar="SERIES/ORDER/MODE=DIR",
                        help="ORDER can be ab, ba, aa or a/b for a single leg; MODE no-profiler, l0, l1 or l1-late")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    schedule = read(args.schedule)
    pairs = [parse_pair(spec, schedule, args.switch) for spec in args.pair]
    rows, variation = summarize(pairs)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    result = {"schedule": str(args.schedule), "schedule_sha256": hashlib.sha256(args.schedule.read_bytes()).hexdigest(),
              "eight_column_names": COLUMNS, "eight_column_rows": rows, "pairs": pairs, "paired_variation": variation}
    (out / "paired_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                              encoding="utf-8")
    (out / "eight_columns.md").write_text(markdown(rows), encoding="utf-8")
    write_tables(out, pairs)
    print(json.dumps({"eight_column_rows": rows, "paired_variation": variation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
