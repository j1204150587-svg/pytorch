"""Run dynamic overflow gates with the existing host88 runner's NPU checks and card lock."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys


STEPS = 8
OVERFLOW_STEPS = [2, 4, 6]
HERE = Path(__file__).resolve().parent


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def env_map(raw):
    value = json.loads(raw)
    if not isinstance(value, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in value.items()):
        raise ValueError("expected a JSON object with string environment names and string values")
    return value


def stage(source, destination):
    shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".git"))
    entry = destination / "train_deepspeed.py"
    code = entry.read_text(encoding="utf-8")
    replacements = [
        ('    metrics = []\n', '    from dynamic_semantic import SemanticAudit\n'
         '    semantic = SemanticAudit(engine, rank_dir / "semantic.json")\n    metrics = []\n'),
        ('                    engine.step()\n',
         '                    semantic.step(step, items, targets, groups, features, loss)\n'),
        # The three deliberately skipped updates do not enter the norm implementation.
        ('finish_norm_run(norm_stats, rank_dir, args.steps, True)',
         'finish_norm_run(norm_stats, rank_dir, args.steps - 3, True)'),
    ]
    for before, after in replacements:
        verify(code.count(before) == 1, f"unexpected dynamic training entry for {before!r}")
        code = code.replace(before, after)
    entry.write_text(code, encoding="utf-8")
    shutil.copyfile(HERE / "dynamic_semantic.py", destination / "dynamic_semantic.py")


def compare(output, schedule, legs):
    records = {}
    libraries = {}
    activation_totals = {}
    for variant, directory in legs.items():
        rank = directory / "rank000"
        values = read(rank / "semantic.json")
        metrics = read(rank / "metrics.json")
        scan = read(rank / "overflow_scan.json")
        verify(len(values) == len(metrics["steps"]) == STEPS and metrics["summary"]["steps"] == STEPS,
               "incomplete eight-step training")
        verify(scan["valid"] and scan["calls"] == scan["fast_calls"] == STEPS and
               scan["fallback_calls"] == scan["errors"] == 0, f"{variant}: 01 overflow activation")
        verify([v["shape"] for v in values] == schedule[:STEPS], f"{variant}: wrong shape schedule")
        verify([v["index"] for v in values if v["overflow"]] == OVERFLOW_STEPS, f"{variant}: overflow pattern")
        for index, (row, metric) in enumerate(zip(values, metrics["steps"])):
            verify(row["index"] == metric["step"] == index and row["loss"] == metric["loss"],
                   f"{variant}: mismatched training log at step {index}")
        activation_path = rank / "native_activation.json"
        if activation_path.is_file():
            evidence = read(activation_path)
            libraries[variant] = evidence["sha256"]
            changes = evidence["per_step"]
            verify([d["shape"] for d in changes] == schedule[:STEPS], f"{variant}: native counter window")
            names = evidence["initial_counters"].keys()
            activation_totals[variant] = {k: sum(d["delta"][k] for d in changes) for k in names}
            env = read(directory / "command.json")["env"]
            if any(env.get(switch) == "1" for switch in ("NPU_DYNAMIC_REBASE_INDICES", "NPU_DYNAMIC_REBASE_FORWARD")):
                totals = activation_totals[variant]
                verify(all(totals.get(k, 0) > 0 for k in ("eligible", "key_uses", "descriptor_uses")),
                       f"{variant}: missing rebase activation {totals}")
                verify(len({tuple(d["shape"]) for d in changes if d["delta"].get("eligible", 0)}) > 1,
                       f"{variant}: rebase activated on fewer than two shapes")
            elif env.get("NPU_DYNAMIC_META_UNFLATTEN") == "1":
                verify([d["delta"]["eligible"] for d in changes] == [1,1,0,1,0,1,0,1],
                       f"{variant}: wrong set of weight updates taking the Meta path")
                verify(activation_totals[variant]["fallback"] >= STEPS, f"{variant}: missing small-list fallback")
            elif evidence["counter"] in ("npu_dynamic_index_rebase_counter", "meta_view_helper"):
                verify(not any(activation_totals[variant].values()), f"{variant}: inactive native switch recorded work")
        records[variant] = values
    if libraries:
        verify(libraries.get("A") == libraries.get("B"), "A/B loaded different native libraries")
    diffs = []
    max_loss_diff = max_norm_diff = 0.0
    for a, b in zip(records["A"], records["B"]):
        for field in a.keys() | b.keys():
            if field in ("norms", "loss"):
                continue
            if a.get(field) != b.get(field):
                diffs.append([a["index"], field])
        max_loss_diff = max(max_loss_diff, abs(a["loss"] - b["loss"]))
        if not math.isclose(a["loss"], b["loss"], rel_tol=1e-6, abs_tol=1e-6):
            diffs.append([a["index"], "loss"])
        if len(a["norms"]) != len(b["norms"]):
            diffs.append([a["index"], "norm_count"])
        for x, y in zip(a["norms"], b["norms"]):
            max_norm_diff = max(max_norm_diff, abs(x - y))
            if not math.isclose(x, y, rel_tol=1e-6, abs_tol=1e-6):
                diffs.append([a["index"], "norm"])
    result = {"passed": not diffs, "differences": diffs, "shape_schedule": schedule[:STEPS],
              "skipped_steps": OVERFLOW_STEPS, "updated_steps": [0, 1, 3, 5, 7],
              "gradients_compared_per_step": [r["gradient_count"] for r in records["A"]],
              "max_absolute_loss_difference": max_loss_diff, "max_absolute_norm_difference": max_norm_diff}
    if libraries:
        result.update(loaded_libtorch_npu_sha256=libraries, native_counter_totals=activation_totals)
    write(output / "comparison.json", result)
    verify(result["passed"], f"semantic A/B mismatch: {diffs}")
    return result


def main():
    import fcntl

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-dir", required=True, type=Path, help="directory containing dynamic run_npu8.py")
    parser.add_argument("--workspace", required=True, type=Path, help="completed staged dynamic eager workspace")
    parser.add_argument("--output", required=True, type=Path, help="new run directory beneath the artifacts bind mount")
    parser.add_argument("--artifacts", default=Path("/data/recsys-gr-ds019-host88/artifacts"), type=Path)
    parser.add_argument("--env-a", type=env_map, default={})
    parser.add_argument("--env-b", type=env_map, required=True)
    parser.add_argument("--common-env", type=env_map, default={})
    parser.add_argument("--order", choices=("ab", "ba"), default="ab")
    args = parser.parse_args()
    sys.path.insert(0, str(args.runner_dir.resolve(strict=True)))
    import run_npu8 as runner

    output, artifacts = args.output.resolve(), args.artifacts.resolve(strict=True)
    source = args.workspace.resolve(strict=True)
    verify(output.is_relative_to(artifacts) and not output.is_relative_to(source), "invalid gate workspace")
    output.mkdir(parents=True, exist_ok=False)
    work = output / "workspace"
    stage(source, work)
    files = [p for p in work.rglob("*") if p.is_file()]
    write(output / "workspace_sha256.json", {str(p.relative_to(work)): sha(p) for p in sorted(files)})
    write(output / "source.json", {"original_train_deepspeed.py": sha(source / "train_deepspeed.py"),
                                   "workspace": str(source), "run_npu8.py": sha(args.runner_dir / "run_npu8.py")})
    def inner(path):
        return str(Path("/artifacts") / path.relative_to(artifacts))

    legs = {}
    with runner.LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        container, init = runner.inspect_container(artifacts, output)
        for number, variant in enumerate(args.order.upper(), 1):
            directory = output / f"{number:02d}-{variant}"
            directory.mkdir()
            legs[variant] = directory
            env = {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
                   **args.common_env, **(args.env_a if variant == "A" else args.env_b),
                   "TASK_QUEUE_ENABLE": "1", "RECSYS_ZERO2_BATCH_OVERFLOW": "1", "RECSYS_DYNAMIC_STATE_AUDIT": "0"}
            command = [*runner.DOCKER, "exec", "--workdir", inner(work),
                       *(x for k, v in env.items() for x in ("--env", f"{k}={v}")),
                       container, "bash", "./run_training.sh", "--model-config", "configs/model_many_tables.json",
                       "--deepspeed-config", "configs/deepspeed_zero2_bf16_overflow_on.json",
                       "--steps", str(STEPS), "--micro-batch-size", "2", "--seed", "1234",
                       "--learning-rate", "0.001", "--weight-decay", "0.01", "--log-interval", "1",
                       "--output-dir", inner(directory), "--shape-schedule", "configs/dynamic_shapes_npu8.json"]
            write(directory / "command.json", {"argv": command, "variant": variant, "env": env})
            runner.check_card(directory, "before", container, init)
            mapping = read(directory / "before.card.json")
            verify((mapping["board"], mapping["chip"], mapping["physical_id"]) == (4, 0, 8),
                   "unexpected logical 8 mapping")
            with (directory / "train.log").open("wb") as stream:
                result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
            (directory / "train.exit").write_text(f"{result.returncode}\n")
            runner.check_card(directory, "after", container, init)
            result.check_returncode()
    result = compare(output, read(work / "configs/dynamic_shapes_npu8.json"), legs)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
