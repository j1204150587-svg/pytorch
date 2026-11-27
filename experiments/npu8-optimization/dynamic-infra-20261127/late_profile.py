"""Prepare a separate copy of the dynamic runner for the Level1 window at steps 12 and 13."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys


def change(text, before, after):
    if text.count(before) != 1:
        raise ValueError(f"unexpected runner source for {before!r}")
    return text.replace(before, after)


def prepare(source, out):
    source, out = source.resolve(strict=True), out.resolve()
    if not (source / "run_suite.py").exists() or out == source or out.is_relative_to(source):
        raise ValueError("source must be the adapted dynamic runner, destination must be separate")
    runner = (source / "run_npu8.py").read_text(encoding="utf-8")
    if "DYNAMIC_INFRA_RECIPE" not in runner:
        raise ValueError("apply adapt_runner.py to the source runner before preparing this window")
    runner = change(runner, '"steps": 8 if profile else args.warmup + args.measure,',
                    '"steps": (16 if mode == "l1" else 8) if profile else args.warmup + args.measure,')
    runner = change(runner, '    for source_file in workspace.glob("*.py"):', '''    if getattr(args, "mode", None) == "l1":
        entry = workspace / "train_deepspeed.py"
        code = entry.read_text(encoding="utf-8")
        for before, after in (("profiler.schedule(wait=2, warmup=2,", "profiler.schedule(wait=10, warmup=2,"),
                              ("args.steps < 4 + args.profile_active", "args.steps < 12 + args.profile_active"),
                              ("profile requires 2 wait + 2 warmup", "profile requires 10 wait + 2 warmup")):
            require(code.count(before) == 1, f"unexpected profile schedule for {before}")
            code = code.replace(before, after)
        entry.write_text(code, encoding="utf-8")
    for source_file in workspace.glob("*.py"):''')
    runner = change(runner, 'validate_step_csv(csv_path, flags["profile_active"])',
                    'validate_step_csv(csv_path, flags["profile_active"], 12 if args.mode == "l1" else 4)')
    evidence = (source / "evidence.py").read_text(encoding="utf-8")
    evidence = change(evidence, "def validate_step_csv(path, active):", "def validate_step_csv(path, active, first=4):")
    evidence = change(evidence, "list(range(4, 4 + active))", "list(range(first, first + active))")
    compile(runner, "run_npu8.py", "exec")
    compile(evidence, "evidence.py", "exec")
    out.mkdir(parents=True, exist_ok=False)
    for path in source.iterdir():
        if path.is_file() and path.suffix in (".py", ".patch", ".json"):
            shutil.copyfile(path, out / path.name)
    (out / "run_npu8.py").write_text(runner, encoding="utf-8")
    (out / "evidence.py").write_text(evidence, encoding="utf-8")
    plan = {"mode": "l1", "steps": 16, "wait": 10, "warmup": 2, "active": [12, 13],
            "source": str(source),
            "sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*"))}}
    (out / "late-window.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    return plan


def run(runner, recipe, order, result):
    runner = runner.resolve(strict=True)
    recipe = recipe.resolve(strict=True)
    config = json.loads(recipe.read_text(encoding="utf-8"))
    if "gate" in config:
        gate = recipe.parent / config["gate"] / "comparison.json"
        if not json.loads(gate.read_text(encoding="utf-8"))["passed"]:
            raise ValueError(f"failed semantic gate {gate}")
    if result.exists():
        raise FileExistsError(f"select a new result path: {result}")
    sys.path.insert(0, str(runner))
    os.environ["DYNAMIC_INFRA_RECIPE"] = str(recipe)
    from run_suite import run_pair
    record = run_pair("NATIVE", "l1", order)
    record.update(recipe=recipe.name, profiler_steps=[12, 13])
    result.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record["exit"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    setup = actions.add_parser("prepare")
    setup.add_argument("--runner", type=Path, required=True)
    setup.add_argument("--out", type=Path, required=True)
    launch = actions.add_parser("run")
    launch.add_argument("--runner", type=Path, required=True)
    launch.add_argument("--recipe", type=Path, required=True)
    launch.add_argument("--order", choices=("ab", "ba", "baseline", "screen"), required=True)
    launch.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "prepare":
        print(json.dumps(prepare(args.runner, args.out)))
    else:
        sys.exit(run(args.runner, args.recipe, args.order, args.result))
