"""Recompute original HSTU steady and Level0 windows, paired by experimental order."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def locate(pair, raw_root):
    if (pair / "plan.json").exists():
        return pair, None
    entry = read(pair / "result.json")
    require(entry["exit"] == 0 and not entry["audit"], f"invalid performance pair {pair}")
    original = Path(entry["directory"])
    raw = raw_root / original.name if raw_root else original
    recipe = read(pair / "recipe.json")
    require("INFRA_COUNTER_DIR" not in recipe["common"], f"audit-only counter active: {pair}")
    for leg in ("01-" + entry["order"][0].upper(), "02-" + entry["order"][1].upper()):
        worker = read(pair / leg / "worker.json")
        value = worker["env"][recipe["switch"]]
        require(value == recipe[leg[-1]] and set(worker["libraries"]) == set(recipe["libraries"]),
                f"native switch or loaded library mismatch: {pair / leg}")
    return raw, recipe


def leg_result(leg, schedule, mode):
    data = read(leg / "rank000/metrics.json")
    rows = data["steps"]
    require([r["step"] for r in rows] == list(range(len(rows))), f"interrupted steps: {leg}")
    shapes = [[r["batch_size"], r["seq_len"]] for r in rows]
    require(shapes == schedule[:len(rows)], f"shape order differs: {leg}")
    if mode == "no-profiler":
        window = data["summary"]["steady_state"]
        samples = sum(b for b, _ in shapes[10:])
        tokens = sum(b * length for b, length in shapes[10:])
        require((window["start_step_inclusive"], window["end_step_exclusive"], len(shapes),
                 window["steps"], samples, tokens) == (10, 60, 60, 50, 136, 2994),
                f"unexpected steady workload: {leg}")
        require((window["samples"], window["tokens"], window["loss_log_policy"]) ==
                (samples, tokens, "deferred_until_after_end_sync"), f"steady accounting differs: {leg}")
        seconds = window["elapsed_s_synchronized"]
        require(math.isfinite(seconds) and seconds > 0, f"invalid time: {leg}")
        return dict(ms_per_step=seconds*1000/50, elapsed_s=seconds, samples=samples, tokens=tokens,
                    samples_per_s=samples/seconds, tokens_per_s=tokens/seconds)
    with (leg / "step_trace_time.csv").open(encoding="utf-8-sig", newline="") as stream:
        trace = list(csv.DictReader(stream))
    require([int(r["Step"]) for r in trace] == [4, 5, 6] and len({r["Device_id"] for r in trace}) == 1,
            f"invalid L0 steps or device: {leg}")
    values = {}
    for name in ("Free", "Stage", "Computing"):
        times = [float(r[name])/1000 for r in trace]
        require(all(math.isfinite(x) and x >= 0 for x in times), f"invalid {name}: {leg}")
        values[name + "_ms_per_step"] = mean(times)
    require(values["Stage_ms_per_step"] > 0, f"empty Stage: {leg}")
    return dict(**values, free_ratio=values["Free_ms_per_step"]/values["Stage_ms_per_step"],
                steps_and_shapes=[[i, *shapes[i]] for i in (4, 5, 6)])


def paired(pair, mode, schedule, raw_root, schedule_digest):
    raw, recipe = locate(pair, raw_root)
    plan, status = read(raw / "plan.json"), read(raw / "status.json")
    require(status["exit"] == 0 and plan["mode"] == mode and not plan["audit_state"]
            and plan["shape_schedule_sha256"] == schedule_digest, f"invalid pair: {raw}")
    order = plan["stage"]
    require(order in ("ab", "ba"), f"invalid pair order: {raw}")
    records = {leg[-1]: leg_result(raw / leg, schedule, mode) for leg in
               ("01-" + order[0].upper(), "02-" + order[1].upper())}
    return {"directory": str(raw), "order": order, "recipe": recipe, **records}


def gain(a, b):
    return None if not a else (1 - b/a) * 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round", nargs=2, metavar=("STEADY", "LEVEL0"), action="append", type=Path, required=True,
                        help="original directories, or host_pair receipt directories; repeat for BA")
    parser.add_argument("--raw-root", type=Path, help="optional downloaded original runs, named by original run ID")
    parser.add_argument("--schedule", type=Path, required=True)
    args = parser.parse_args()
    reference = read(Path(__file__).with_name("reference.json"))
    schedule_digest = hashlib.sha256(args.schedule.read_bytes()).hexdigest()
    require(schedule_digest == reference["workspace"]["configs/dynamic_shapes_npu8.json"], "schedule digest differs")
    shape_list = read(args.schedule)
    results, step_deltas, recipes = [], [], []
    for steady, level0 in args.round:
        timing = paired(steady, "no-profiler", shape_list, args.raw_root, schedule_digest)
        profile = paired(level0, "l0", shape_list, args.raw_root, schedule_digest)
        require(timing["order"] == profile["order"] and timing["recipe"] == profile["recipe"],
                "pair the same order, switches and native package")
        recipes.append(timing["recipe"])
        a, b = timing["A"]["ms_per_step"], timing["B"]["ms_per_step"]
        fa, fb = profile["A"]["Free_ms_per_step"], profile["B"]["Free_ms_per_step"]
        ra, rb = profile["A"]["free_ratio"], profile["B"]["free_ratio"]
        step_deltas.append(a-b)
        results.append(dict(order=timing["order"], steady=timing, level0=profile, delta_ms_per_step=a-b,
                            eight_columns={"baseline_ms":a, "candidate_ms":b, "step_gain_pct":gain(a,b),
                                           "baseline_free_ms":fa, "baseline_free_pct":ra*100, "candidate_free_ms":fb,
                                           "candidate_free_pct":rb*100, "free_ratio_gain_pct":gain(ra,rb)}))
    require(all(r == recipes[0] for r in recipes), "different native switches or libraries across rounds")
    print(json.dumps({"rounds": results, "paired_delta_ms_per_step": {"count": len(results), "mean": mean(step_deltas),
                      "min": min(step_deltas), "max": max(step_deltas),
                      "sample_sd": stdev(step_deltas) if len(results) > 1 else None}}, indent=2))


if __name__ == "__main__":
    main()
