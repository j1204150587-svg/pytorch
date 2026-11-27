"""Run guarded native microchecks with the same host lock and device checks as training."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runner"))
import run_npu8 as runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.recipe.read_text())
    out = args.output.resolve()
    out.mkdir(exist_ok=False)
    artifacts = HERE.parent
    def inner(path):
        return "/artifacts/" + path.relative_to(artifacts).as_posix()
    with runner.LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        container, init = runner.inspect_container(artifacts, out)
        for idx, variant in enumerate(("A", "B"), 1):
            leg = out / f"{idx:02d}-{variant}"
            leg.mkdir()
            env = {**cfg["common"], **cfg[variant], "TASK_QUEUE_ENABLE":"1","PYTHONDONTWRITEBYTECODE":"1"}
            source = "rebase_forward" if "NPU_DYNAMIC_REBASE_FORWARD" in env else "rebase"
            cmd = [*runner.DOCKER,"exec", *(x for k,v in env.items() for x in ("--env",k+"="+v)),
                   container,"python3", "-B", inner(HERE / source / "micro_gate.py"), inner(leg / "micro.json")]
            (leg/"command.json").write_text(json.dumps(cmd))
            runner.check_card(leg,"before",container,init)
            with (leg/"run.log").open("wb") as log:
                rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
            runner.check_card(leg,"after",container,init)
            if rc: raise SystemExit(rc)

if __name__ == "__main__":
    main()

