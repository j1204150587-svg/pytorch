"""Run #44 A/B micro gates under the existing host card lock and check the loaded library."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--runner-dir", type=Path, required=True)
    cli.add_argument("--native-dir", type=Path, required=True, help="completed build_copy.py output")
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument("--artifacts", type=Path, default=Path("/data/recsys-gr-ds019-host88/artifacts"))
    args = cli.parse_args()
    sys.path.insert(0, str(args.runner_dir.resolve(strict=True)))
    import run_npu8 as r
    artifacts, out = args.artifacts.resolve(strict=True), args.out.resolve()
    def inner(path):
        return str(Path("/artifacts") / path.resolve().relative_to(artifacts))
    gate = inner(Path(__file__).resolve().with_name("gate.py"))
    package = inner(args.native_dir / "package")
    expected = json.loads((args.native_dir / "build-receipt.json").read_text())["library_sha256"]
    out.mkdir(parents=True, exist_ok=False)
    with r.LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        container, init = r.inspect_container(artifacts, out)
        for idx, variant in enumerate(("A", "B"), 1):
            leg = out / f"0{idx}-{variant}"
            leg.mkdir()
            env = {"PYTHONPATH": package, "TASK_QUEUE_ENABLE": "1", "NPU_EXPERIMENT_GUARDED_D2D": "1",
                   "NPU_EXPERIMENT_D2D_RECORD_SRC_MARK": "1", "NPU_EXPERIMENT_D2D_RECORD_SRC": str(idx - 1),
                   "PYTHONDONTWRITEBYTECODE": "1"}
            cmd = [*r.DOCKER, "exec", "-w", "/workspace",
                   *(v for k, value in env.items() for v in ("--env", f"{k}={value}")), container, "bash", "-c",
                   'set -eo pipefail; source ./env.sh; unset ASCEND_RT_VISIBLE_DEVICES; exec "$@"', "gate",
                   "python3", "-B", gate, variant, inner(leg / "micro.json")]
            (leg / "command.json").write_text(json.dumps({"argv": cmd, "env": env}, indent=2))
            r.check_card(leg, "before", container, init)
            with (leg / "train.log").open("wb") as stream:
                proc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT)
            (leg / "train.exit").write_text(str(proc.returncode) + "\n")
            r.check_card(leg, "after", container, init)
            proc.check_returncode()
            got = json.loads((leg / "micro.json").read_text())
            if got["sha256"] != expected or got["status"] != "passed":
                raise ValueError(f"{variant}: wrong library or failed micro gate")
    (out / "result.json").write_text(json.dumps({"passed": True, "sha256": expected, "order": "AB"}, indent=2))


if __name__ == "__main__":
    main()
