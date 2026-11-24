"""Validate actual outputs from the 26 state-ownership A/B launcher."""

import argparse
import csv
import json
import math
from pathlib import Path


CASES = ("initial_nan", "finite_below_clip", "nan", "posinf", "neginf",
         "finite_recovery_above_clip", "finite_raw", "finite_raw_next")
LOSS_TOL = {"rel_tol": 1e-6, "abs_tol": 1e-7}
NORM_TOL = {"rel_tol": 1e-5, "abs_tol": 1e-7}


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate_leg(rank, letter, mode):
    def report(name):
        return read(rank / f"{name}.json")

    steady = mode == "steady"
    steps = 60 if steady else 8
    applied = 4 if mode == "correctness" else steps
    metrics = report("metrics")
    summary = metrics["summary"]
    require(summary["status"] == "completed" and summary["steps"] == steps
            and summary["profile_enabled"] is (mode == "level0")
            and summary["overlap_input"] is True
            and summary["overlap_input_hits"] == steps - 1 - int(steady), f"{letter}: wrong training work")
    graph = report("hstu_graph")
    require(graph == {"parameter_aliases": 1052, "counts": {"forward": steps, "backward": steps},
                      "expected_steps": steps, "sidepack": True, "direct_gradient": True,
                      "direct_gradient_hits": steps}, f"{letter}: replay or gradient hits differ")
    scan = report("overflow_scan")
    norm = report("norm_stats")
    require(scan["valid"] and scan["calls"] == scan["fast_calls"] == scan["chunks"] == steps
            and scan["fallback_calls"] == scan["errors"] == 0 and scan["first_npu_formats"] == {"2": 1052}
            and norm["valid"] and norm["calls"] == norm["batched_calls"] == applied
            and not norm["fallback_reasons"], f"{letter}: missing PTA statistics or used fallback")
    require(report("pta_wrapper") == {"scope": "both", "graph": True, "norm": applied, "overflow": steps},
            f"{letter}: wrong PTA graph replay count")
    if letter == "A":
        require(report("fast_partition") == {"partition": steps}
                and report("zero2_local_step") == {"mode": "both", "flatten": applied, "aliases": applied}
                and not (rank / "state_owner.json").exists(), "A: missing baseline 09 step")
    else:
        owner = report("state_owner")
        require(owner["owner"] == "single_rank_executor" and owner["consumed"] == owner["epilogue_hits"] == steps
                and owner["parameters"] == 1052 and owner["flat_numel"] == 4_380_544
                and owner["counts"] == {"global_steps": steps, "micro_steps": steps, "gas_boundary_ctr": steps,
                                        "global_samples": steps * 2, "skipped_steps": steps - applied,
                                        "successful_updates": applied}
                and not (rank / "fast_partition.json").exists()
                and not (rank / "zero2_local_step.json").exists(), "B: wrong owner or update counts")
    losses = {r["step"]: r["loss"] for r in metrics["steps"] if "loss" in r}
    require([row["step"] for row in metrics["steps"]] == list(range(steps))
            and all(math.isfinite(loss) for loss in losses.values())
            and list(losses) == ([0, 9, 19, 29, 39, 49, 59] if steady else [0, 3, 7]),
            f"{letter}: incomplete loss series")
    args = report("run_arguments")
    del args["output_dir"]
    runtime = report("runtime")
    require((runtime["rank"], runtime["world_size"], runtime["local_rank"], runtime["deepspeed"])
            == (0, 1, 0, "0.19.0"), f"{letter}: unexpected runtime")
    return {"runtime": runtime, "args": args, "config": report("effective_deepspeed_config"),
            "model_summary": report("model_summary"), "losses": losses, "summary": summary,
            "norm_layout_counts": [norm["batches"], norm["batched_gradients"]]}


def check_gate(path, bundle):
    gate = read(path)
    require(gate["kind"] == "26_integrated_correctness" and gate["status"] == "passed"
            and gate["order"] == "ab" and tuple(gate["cases"]) == CASES
            and gate["bundle_sha256"] == bundle, "require passing integration correctness on this bundle")
    return gate


def run(mode, order, output, bundle, gate_path):
    ranks = [output / f"{i:02d}-{letter.upper()}/rank000" for i, letter in enumerate(order, 1)]
    a, b = [validate_leg(rank, letter.upper(), mode) for rank, letter in zip(ranks, order, strict=True)]
    for key in ("runtime", "args", "config", "model_summary", "norm_layout_counts"):
        require(a[key] == b[key], f"A/B has differing {key}")
    for step, loss in a["losses"].items():
        require(math.isclose(loss, b["losses"][step], **LOSS_TOL), f"loss differs at step {step}")
    launch = dict(row.split("=", 1) for row in (output / "launch.txt").read_text().splitlines())
    environment = {key: launch[key] for key in ("TASK_QUEUE_ENABLE", "RECSYS_NPU_IDS")}
    result = {"status": "passed", "kind": f"26_integrated_{mode}", "order": order, "bundle_sha256": bundle,
              "loss_tolerance": LOSS_TOL, "runtime": a["runtime"], "environment": environment}
    if mode == "correctness":
        reports = [read(rank / "integration_correctness.json") for rank in ranks]
        inputs = [read(rank / "overlap_input_audit.json") for rank in ranks]
        require(inputs[0] == inputs[1] and len(inputs[0]["input_sha256"]) == len(CASES),
                "batch input or RNG digests differ")
        require(reports[0]["layout"] == reports[1]["layout"]
                and any(entry[0] == ["item_embedding.weight", "output.weight"] for entry in reports[0]["layout"]),
                "parameter map or tied alias differs")
        for report, letter in zip(reports, ("A", "B"), strict=True):
            require(report["status"] == "passed" and report["owner"] == letter
                    and tuple(row["case"] for row in report["steps"]) == CASES, "missing integrated case coverage")
        for x, y in zip(reports[0]["steps"], reports[1]["steps"], strict=True):
            case = x["case"]
            require(x.keys() == y.keys(), f"{case}: report fields differ")
            for key in x:
                tolerance = {"scaled_norm": NORM_TOL, "loss": LOSS_TOL}.get(key)
                equal = math.isclose(x[key], y[key], **tolerance) if tolerance else x[key] == y[key]
                require(equal, f"{case}: {key} differs")
        result.update(cases=CASES, norm_tolerance=NORM_TOL,
                      state_comparison="bitwise", raw_gradient_comparison="bitwise")
    else:
        gate = check_gate(gate_path, bundle)
        require(gate["runtime"] == a["runtime"] and gate["environment"] == environment,
                "timing runtime or queue/card selection differs from the integration gate")
        require(all(not (rank / "integration_correctness.json").exists()
                    and not (rank / "overlap_input_audit.json").exists() for rank in ranks),
                "correctness instrumentation present in measurement output")
        if mode == "steady":
            measured = []
            for leg in (a, b):
                steady = leg["summary"]["steady_state"]
                require([steady[key] for key in ("start_step_inclusive", "end_step_exclusive", "steps", "samples")]
                        == [10, 60, 50, 100] and steady["loss_log_policy"] == "deferred_until_after_end_sync"
                        and math.isfinite(steady["elapsed_s_synchronized"]) and steady["elapsed_s_synchronized"] > 0
                        and math.isclose(steady["ms_per_step"], steady["elapsed_s_synchronized"] * 1000 / 50),
                        "invalid synchronized steady measurement window")
                measured.append(steady)
            result["recorded_windows"] = measured
        else:
            trace_artifacts = []
            for rank in ranks:
                traces = list((rank / "profile").rglob("trace_view.json"))
                require(len(traces) == 1 and traces[0].stat().st_size > 0, "missing Level0 raw trace")
                csv_path = traces[0].with_name("step_trace_time.csv")
                with csv_path.open(newline="", encoding="utf-8-sig") as stream:
                    rows = list(csv.DictReader(stream))
                require([int(row["Step"]) for row in rows] == [4, 5, 6]
                        and all(row["Device_id"] == "0" for row in rows), "unexpected Level0 steps")
                trace_artifacts.append({"trace": str(traces[0].relative_to(output)),
                                        "step_csv": str(csv_path.relative_to(output))})
            result["trace_artifacts"] = trace_artifacts
    (output / "comparison.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(output / "comparison.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("correctness", "steady", "level0", "check-gate"))
    parser.add_argument("path", type=Path)
    parser.add_argument("bundle_sha256")
    parser.add_argument("--order", choices=("aa", "ab", "ba"))
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    if args.mode == "check-gate":
        check_gate(args.path, args.bundle_sha256)
    else:
        require(args.order is not None and (args.mode != "correctness" or args.order == "ab"), "invalid run order")
        require((args.gate is None) == (args.mode == "correctness"), "require --gate for a timing comparison")
        run(args.mode, args.order, args.path, args.bundle_sha256, args.gate)


if __name__ == "__main__":
    main()
