"""Collect a microbenchmark or one host call profile with the existing NPU 8 reservation."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
ARTIFACTS = STUDY.parent
sys.path.insert(0, str(STUDY / "runner"))
import run_npu8 as runner


def inside(path):
    return str(Path("/artifacts") / path.resolve().relative_to(ARTIFACTS))


def host_profile(workspace, output):
    work = output / "workspace"
    shutil.copytree(workspace, work, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    trainer = work / "train_deepspeed.py"
    source = trainer.read_text()
    start = "                step_start = time.perf_counter()"
    stop = "                metrics.append(metric)"
    if source.count(start) != 1 or source.count(stop) != 1:
        raise ValueError("unexpected training loop")
    source = source.replace(start, """                if step == 4:
                    import cProfile
                    host_profiler = cProfile.Profile()
                    host_profiler.enable()
""" + start)
    source = source.replace(stop, stop + """
                if step == 6:
                    host_profiler.disable()
                    host_profiler.dump_stats(str(rank_dir / 'steps-4-6.prof'))
""")
    trainer.write_text(source)
    (output / "diagnostic.sha256").write_text(hashlib.sha256(source.encode()).hexdigest())
    return ["--workdir", inside(work)], [
        "bash", "./run_training.sh", "--steps", "8", "--micro-batch-size", "2",
        "--seed", "1234", "--learning-rate", "0.001", "--weight-decay", "0.01",
        "--log-interval", "4", "--model-config", "configs/model_many_tables.json",
        "--deepspeed-config", "configs/deepspeed_zero2_bf16_overflow_on.json",
        "--shape-schedule", "configs/dynamic_shapes_npu8.json", "--output-dir", inside(output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--recipe", type=Path)
    parser.add_argument("--variant", choices=("A", "B"), default="B")
    parser.add_argument("--script", type=Path)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    env = {"TASK_QUEUE_ENABLE": "1", "RECSYS_ZERO2_BATCH_OVERFLOW": "1", "OMP_NUM_THREADS": "2",
           "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1"}
    if args.recipe:
        recipe = json.loads(args.recipe.read_bytes())
        env.update(recipe["common"])
        env.update(recipe[args.variant])
    if args.workspace:
        setup, argv = host_profile(args.workspace, out)
    else:
        values = args.args[1:] if args.args[:1] == ["--"] else args.args
        values = [inside(out / "micro.json") if v == "{output}" else v for v in values]
        setup, argv = [], ["bash", "-c", 'source /workspace/env.sh && unset ASCEND_RT_VISIBLE_DEVICES && exec python3 -B "$@"',
                           "micro", inside(args.script), *values]
    with runner.LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        container, init = runner.inspect_container(ARTIFACTS, out)
        runner.check_card(out, "before", container, init)
        command = [*runner.DOCKER, "exec", *setup, *(part for k,v in env.items() for part in ("--env", k+"="+v)),
                   container, *argv]
        (out / "command.json").write_text(json.dumps(command, indent=2))
        with (out / "run.log").open("wb") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        (out / "train.exit").write_text(str(result.returncode))
        runner.check_card(out, "after", container, init)
        result.check_returncode()


if __name__ == "__main__":
    main()
