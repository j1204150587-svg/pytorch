"""Run one default-capacity, first-eight-then-same-eight cache-key diagnostic."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from summarize_cache import main as summarize


def repeat_schedule(destination):
    schedule = destination / "dynamic_shapes_npu8.json"
    original = json.loads(schedule.read_text())
    if len(original) != 60 or len(set(map(tuple, original))) != 60:
        raise ValueError("expected the standard sequence of 60 distinct shapes")
    first_eight = original[:8]
    schedule.write_text(json.dumps(first_eight * 2, indent=2) + "\n")
    return schedule


def prepare(runner, destination):
    runner = runner.resolve(strict=True)
    base = (runner / "run_npu8.py").read_text()
    if "DYNAMIC_INFRA_RECIPE" not in base:
        raise ValueError("expected the adapted, shared-lock training runner")
    for item in runner.iterdir():
        if item.is_file() and item.suffix in {".py", ".json", ".patch"}:
            shutil.copyfile(item, destination / item.name)
    schedule = repeat_schedule(destination)
    suite = destination / "run_suite.py"
    code = suite.read_text()
    anchor = "    print(f'WAIT {label}', flush=True)\n"
    if code.count(anchor) != 1:
        raise ValueError("unexpected run_pair entry")
    code = code.replace(anchor, '    command += ["--warmup", "2", "--measure", "14"]\n' + anchor)
    compile(code, str(suite), "exec")
    suite.write_text(code)
    return schedule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--late-level1", action="store_true",
                        help="collect native Level1 for repeated #12/13 without audit, paired to original #4/5")
    args = parser.parse_args()
    study = Path(__file__).resolve().parents[1]
    source = study / "runner"
    stage_root = Path(tempfile.mkdtemp(prefix="repeat-diagnostic-runner-", dir=study))
    if args.late_level1:
        sys.path.insert(0, str(study))
        from late_profile import prepare as prepare_late
        stage = stage_root / "late"
        prepare_late(source, stage)
        schedule = repeat_schedule(stage)
        recipe = (study / "cache-limit.json").resolve(strict=True)
        mode = "l1"
    else:
        stage = stage_root
        schedule = prepare(source, stage)
        recipe = (study / "cache-diag.json").resolve(strict=True)
        mode = "no-profiler"
    config = json.loads(recipe.read_text())
    if bool(config.get("diagnostic")) == args.late_level1 or config.get("A") or "ACLNN_CACHE_LIMIT" in config["common"]:
        raise ValueError("expected a default-capacity A leg, with audit only in no-profiler diagnostic mode")
    os.environ["DYNAMIC_INFRA_RECIPE"] = str(recipe)
    sys.path.insert(0, str(stage))
    from run_suite import run_pair
    record = run_pair("NATIVE", mode, "baseline")
    record["repeat_schedule"] = str(schedule)
    receipt = stage / "repeat-run.json"
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    if record["exit"]:
        return record["exit"]
    session = Path(record["directory"])
    leg = session / "01-A"
    if not args.late_level1:
        summarize(leg, schedule, expected_steps=16)
    shutil.copyfile(receipt, session / "repeat-run.json")
    print(json.dumps({"session": str(session), "receipt": str(receipt), "mode": mode}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
