"""Build a fresh scheduler-only Level1 runner and execute its baseline leg."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys


HERE = Path(__file__).resolve().parent


def change(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"unexpected source at {old!r}")
    return source.replace(old, new)


def install(workspace, runner):
    path = workspace / "train_deepspeed.py"
    code = path.read_text()
    anchor = "        with profiler_context as profiler:\n"
    code = change(code, anchor, """        from train_probe import prepare as sched_prepare, clock_anchors
        sched_dir = rank_dir / "queue_sched"
        sched_prepare(sched_dir)
""" + anchor)
    anchor = "            for step, (batch_size, sequence_length) in enumerate(shapes):\n"
    code = change(code, anchor, anchor + "                clock_anchors(sched_dir, record_function, step)\n")
    end = "        torch.npu.synchronize()\n        end = time.perf_counter()"
    code = change(code, end, '        (sched_dir / "finished").touch()\n' + end)
    compile(code, str(path), "exec")
    path.write_text(code)
    shutil.copyfile(runner / "train_probe.py", workspace / "train_probe.py")


def setup(src, dest):
    if dest.exists():
        raise FileExistsError(dest)
    path = src / "run_npu8.py"
    code = path.read_text()
    anchor = '    for source_file in workspace.glob("*.py"):'
    code = change(code, anchor, '    from sched_runner import install\n    install(workspace, HERE)\n' + anchor)
    call = "completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False, start_new_session=True)"
    code = change(code, call, "from capture import run_target\n            completed = run_target(argv, log, out, workspace, DOCKER, container)")
    compile(code, str(path), "exec")
    dest.mkdir(parents=True)
    for file in src.iterdir():
        if file.is_file() and file.suffix in {".py", ".patch", ".json"}:
            shutil.copyfile(file, dest / file.name)
    (dest / path.name).write_text(code)
    shutil.copyfile(__file__, dest / "sched_runner.py")
    for name in ("train_probe.py", "capture.py"):
        shutil.copyfile(HERE / name, dest / name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()
    if args.result.exists() or os.environ.get("DYNAMIC_INFRA_RECIPE"):
        raise SystemExit("need fresh output and stock default environment")
    setup(args.runner.resolve(strict=True), args.private.resolve())
    if not args.setup_only:
        sys.path.insert(0, str(args.private.resolve()))
        from run_suite import run_pair
        result = run_pair("PARTITION", "l1", "baseline")
        args.result.write_text(json.dumps(result, indent=2) + "\n")
        raise SystemExit(result["exit"])
