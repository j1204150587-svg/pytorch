"""Recompute paired NPU 8 measurements from locally copied runner artifacts."""

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


ORDERS = {"aa": ["A", "A"], "ab": ["A", "B"], "ba": ["B", "A"]}
PROFILE_STEPS = {"l0": [4, 5, 6], "l1": [4, 5]}
MICRO_BATCH = 2  # The new runner uses a single rank with micro batch 2.
CSV_FIELDS = {
    "stage_ms": "Stage",
    "free_ms": "Free",
    "computing_ms": "Computing",
    "unoverlapped_communication_ms": "Communication(Not Overlapped)",
}
GROUP_NAME = re.compile(r"\d{8}T\d{6}Z-(aa|ab|ba|baseline)-(no-profiler|l0|l1)-.+")
INPUT_ERRORS = (OSError, ValueError, KeyError, TypeError, OverflowError, csv.Error)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_object(path):
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(document, dict), "expected a JSON object")
    return document


def nonnegative(value, field):
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            f"invalid {field}: {value!r}")
    return value


def distribution(values):
    ordered = sorted(values)

    def percentile(fraction):
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {"n": len(ordered), "min": ordered[0], "mean": mean(ordered),
            "p50": percentile(0.50), "p90": percentile(0.90), "p99": percentile(0.99),
            "max": ordered[-1]}


def training(path, mode):
    document = read_object(path)
    summary, rows = document["summary"], document["steps"]
    total = summary["steps"]
    require(summary["status"] == "completed" and type(summary["rank"]) is int and summary["rank"] == 0,
            "expected completed rank 0 metrics")
    require(summary["profile_enabled"] is (mode != "no-profiler"), "profiler flag differs from plan")
    require(type(total) is int and total > 0 and isinstance(rows, list) and len(rows) == total,
            "missing or extra training steps")
    require(all(type(row["step"]) is int and row["step"] == step for step, row in enumerate(rows)),
            "unexpected training step IDs")
    return summary, rows


def steady_state(summary, rows):
    raw = summary["steady_state"]
    start, end, steps, samples = (raw[name] for name in
                                 ("start_step_inclusive", "end_step_exclusive", "steps", "samples"))
    require(all(type(value) is int for value in (start, end, steps, samples))
            and 0 < start < end == len(rows) and steps == end - start and samples == steps * MICRO_BATCH,
            "inconsistent steady-state window or sample count")
    require(raw["loss_log_policy"] == "deferred_until_after_end_sync", "unexpected loss logging policy")
    elapsed = nonnegative(raw["elapsed_s_synchronized"], "steady-state elapsed_s_synchronized")
    total_elapsed = nonnegative(summary["elapsed_s_synchronized"], "training elapsed_s_synchronized")
    require(0 < elapsed <= total_elapsed, "steady-state elapsed must be positive and within training time")
    values = {"samples_per_s": samples / elapsed, "ms_per_step": (elapsed / steps) * 1000}
    for key, value in values.items():
        require(math.isfinite(value) and math.isclose(nonnegative(raw[key], key), value),
                f"reported {key} disagrees with synchronized time and counts")
    host_times = [nonnegative(row["host_step_ms"], f"step {row['step']} host_step_ms") for row in rows[start:end]]
    return {
        "window": {"start_step_inclusive": start, "end_step_exclusive": end,
                   "steps": steps, "samples": samples},
        "synchronized_elapsed_s": elapsed,
        "synchronized_window_count": 1,
        "values": values,
        "distributions": {"async_host_step_ms": distribution(host_times)},
    }


def profile(path, mode):
    expected_steps = PROFILE_STEPS[mode]
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, strict=True)
        columns = reader.fieldnames or []
        required = {"Step", "Device_id", *CSV_FIELDS.values()}
        require(required.issubset(columns) and len(columns) == len(set(columns)),
                "missing or duplicate profile CSV columns")
        raw = list(reader)
    require([int(row["Step"]) for row in raw] == expected_steps,
            f"expected {mode} step IDs {expected_steps}")
    device_ids = {(row["Device_id"] or "").strip() for row in raw}
    require(len(device_ids) == 1 and "" not in device_ids, "expected one identified device in profile CSV")
    rows = []
    for step, row in zip(expected_steps, raw):
        times = {key: nonnegative(float(row[column]), f"step {step} {column}") / 1000
                 for key, column in CSV_FIELDS.items()}
        require(0 <= times["free_ms"] <= times["stage_ms"] and times["stage_ms"] > 0,
                f"step {step} requires 0 <= Free <= Stage and positive Stage")
        rows.append({"step": step, **times, "free_over_stage_pct": (times["free_ms"] / times["stage_ms"]) * 100})
    distributions = {key: distribution([row[key] for row in rows])
                     for key in (*CSV_FIELDS, "free_over_stage_pct")}
    values = {key: distributions[key]["mean"] for key in CSV_FIELDS}
    values["free_over_stage_pct"] = (math.fsum(row["free_ms"] for row in rows)
                                     / math.fsum(row["stage_ms"] for row in rows)) * 100
    return {
        "window": {"step_ids": expected_steps, "device_id": next(iter(device_ids)),
                   "steps": len(rows), "samples": len(rows) * MICRO_BATCH},
        "profile_steps": rows,
        "values": values,
        "distributions": distributions,
    }


def read_leg(path, position, variant, mode):
    metrics = path / "rank000" / "metrics.json"
    step_csv = path / "step_trace_time.csv"
    result = {"position": position, "variant": variant, "path": str(path),
              "metrics_path": str(metrics), "profile_csv_path": None, "issues": [], "values": None}
    if mode != "no-profiler":
        result["profile_csv_path"] = str(step_csv)
    try:
        summary, rows = training(metrics, mode)
        result["training_steps"] = len(rows)
        if mode == "no-profiler":
            result.update(steady_state(summary, rows))
        else:
            require(len(rows) == 8 and "steady_state" not in summary, "unexpected profiled training summary")
    except INPUT_ERRORS as error:
        result["issues"].append(f"rank000/metrics.json: {error}")
    if mode != "no-profiler":
        try:
            result.update(profile(step_csv, mode))
        except INPUT_ERRORS as error:
            result["issues"].append(f"step_trace_time.csv: {error}")
    return result


def read_group(path):
    group = {"path": str(path), "stage": None, "mode": None, "order": None, "runner_exit": None,
             "status": "incomplete", "issues": [], "legs": [], "comparison": None}
    try:
        plan = read_object(path / "plan.json")
        if plan.get("stage") == "baseline":
            return None
        stage, mode, order = plan["stage"], plan["mode"], plan["order"]
        require(stage in ORDERS and order == ORDERS[stage], "expected aa, ab or ba with matching order")
        require(mode in ("no-profiler", *PROFILE_STEPS), "unknown runner mode")
        group.update(stage=stage, mode=mode, order=order)
        group["candidate"] = {key: plan[key] for key in ("candidate_patch_sha256", "candidate_module_sha256")
                              if isinstance(plan.get(key), str)}
    except INPUT_ERRORS as error:
        group["issues"].append(f"plan.json: {error}")
        return group

    try:
        group["runner_exit"] = int((path / "runner.exit").read_text(encoding="utf-8-sig").strip())
        require(group["runner_exit"] == 0, f"runner exited with {group['runner_exit']}")
    except INPUT_ERRORS as error:
        group["issues"].append(f"runner.exit: {error}")

    group["legs"] = [read_leg(path / f"{position:02d}-{variant}", position, variant, mode)
                     for position, variant in enumerate(order, 1)]
    first, second = group["legs"]
    if first["issues"] or second["issues"]:
        group["issues"].append("one or more legs have incomplete or invalid inputs")
    elif first["window"] != second["window"]:
        group["issues"].append("legs have different measurement windows or device IDs")
    if group["issues"]:
        return group

    reference, compared = (second, first) if stage == "ba" else (first, second)
    metrics = {}
    for key, reference_value in reference["values"].items():
        delta = compared["values"][key] - reference_value
        metrics[key] = {"delta": delta, "relative_pct": delta / reference_value * 100 if reference_value else None}
    group.update(status="complete", comparison={"reference_position": reference["position"],
                                                 "compared_position": compared["position"], "metrics": metrics})
    return group


def add_natural_variation(groups):
    baselines = [group for group in groups if group["stage"] == "aa" and group["status"] == "complete"]
    for group in groups:
        if group["stage"] not in ("ab", "ba") or group["status"] != "complete":
            continue
        comparison = group["comparison"]
        matched = [aa for aa in baselines if aa["mode"] == group["mode"]
                   and aa["legs"][0]["window"] == group["legs"][0]["window"]]
        comparison["natural_variation"] = {"n_aa_pairs": len(matched), "aa_groups": [aa["path"] for aa in matched]}
        for key, metric in comparison["metrics"].items():
            aa_metrics = [aa["comparison"]["metrics"][key] for aa in matched]
            variation = {}
            for field in ("delta", "relative_pct"):
                magnitudes = [abs(sample[field]) for sample in aa_metrics if sample[field] is not None]
                limit = max(magnitudes, default=None)
                exceeds = None
                if limit is not None and metric[field] is not None:
                    magnitude = abs(metric[field])
                    exceeds = magnitude > limit and not math.isclose(magnitude, limit)
                variation[field] = {"n_aa_pairs": len(magnitudes), "max_absolute": limit,
                                    "exceeds_observed_aa_max_absolute": exceeds}
            metric["natural_variation"] = variation


def looks_like_group(path):
    return (GROUP_NAME.fullmatch(path.name) is not None or (path / "plan.json").is_file()
            or (path / "runner.exit").is_file()
            or any((path / name).is_dir() for name in ("01-A", "01-B", "02-A", "02-B")))


def discover(inputs):
    groups = set()
    for source in inputs:
        path = source.resolve()
        require(path.is_dir(), f"not a directory: {path}")
        if looks_like_group(path):
            groups.add(path)
        else:
            groups.update(child.resolve() for child in path.iterdir() if child.is_dir() and looks_like_group(child))
    return sorted(groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, metavar="DIR",
                        help="runner group directories or their direct parents; single-leg baselines are skipped")
    parser.add_argument("-o", "--output", type=Path, help="write UTF-8 JSON to this path (default: stdout)")
    args = parser.parse_args()
    try:
        groups = [group for path in discover(args.inputs) if (group := read_group(path)) is not None]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    if not groups:
        parser.error("no paired runner groups found in the supplied directories")
    add_natural_variation(groups)
    report = {
        "method": {
            "steady_state": "samples/s = samples / synchronized seconds; ms/step = synchronized seconds * 1000 / steps. Async host_step_ms describes host timing.",
            "profile": "Official CSV durations in us converted to ms; compare means over sampled steps. Group Free/Stage is 100 * sum(Free) / sum(Stage).",
            "distributions": "Unweighted per-step observations; percentiles use linear interpolation. Throughput and synchronized ms/step each describe one window per leg.",
            "delta": "A/A = second A minus first A. A/B and B/A = B minus A. Free/Stage delta is in percentage points.",
            "relative_pct": "100 * delta / reference, where reference is first A in A/A and A in either candidate order. Null when the reference is zero.",
            "natural_variation": "Descriptive maximum absolute A/A difference among complete pairs of the same mode, window and profile device ID. Null when no compatible estimate exists.",
            "completeness": "Pair deltas require runner.exit=0 and both complete, aligned legs. Target-window trace analysis is separate.",
        },
        "groups": groups,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        try:
            args.output.write_text(payload, encoding="utf-8")
        except OSError as error:
            parser.error(str(error))


if __name__ == "__main__":
    main()
