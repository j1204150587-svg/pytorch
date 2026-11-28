"""Run an isolated operator probe using the archived NPU8 ownership checks."""
import argparse
import fcntl
import json
import subprocess
import sys
from pathlib import Path

ART = Path("/data/recsys-gr-ds019-host88/artifacts")
SOURCE = ART / "dynamic-shapes-20261127/deploy.hFNaMd"
WORK = "/artifacts/dynamic-shapes-20261127/20261127T025520Z-ab-no-profiler-2sqpl881/workspace"
PREV = "/artifacts/dynamic-infra-20261127/dynamic-free70"
sys.path.insert(0, str(SOURCE))
import run_npu8 as runner

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("program")
    parser.add_argument("--package")
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("options", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "train.exit").exists():
        raise RuntimeError("existing run")
    opts = [o.replace("@OUT", "/artifacts/" + str(out.relative_to(ART)))
            .replace("@WORK", WORK).replace("@PREV", PREV) for o in args.options]
    with runner.LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        cid, init = runner.inspect_container(ART, out)
        runner.check_card(out, "before", cid, init)
        env = {"TASK_QUEUE_ENABLE": "1", "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
               "PYTHONDONTWRITEBYTECODE": "1"}
        if args.package:
            env["PYTHONPATH"] = args.package
        for setting in args.env:
            name, value = setting.split("=", 1)
            if name not in {"NPU_DYNAMIC_REBASE_INDICES", "NPU_DYNAMIC_REBASE_FORWARD",
                            "NPU_DYNAMIC_REBASE_COUNT", "ACLNN_CACHE_LIMIT"}:
                raise ValueError(name)
            env[name] = value
        cmd = [*runner.DOCKER, "exec", *(i for k,v in env.items() for i in ("--env", k+"="+v)),
               cid, "bash", "-c", 'source "$1/env.sh" && shift && unset ASCEND_RT_VISIBLE_DEVICES && exec python3 -B "$@"',
               "probe", WORK, args.program, *opts]
        (out/"command.json").write_text(json.dumps(cmd, indent=2))
        with (out/"run.log").open("wb") as log:
            rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT).returncode
        (out/"train.exit").write_text(str(rc))
        runner.check_card(out, "after", cid, init)
        return rc

if __name__ == "__main__":
    sys.exit(main())

