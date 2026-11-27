"""Execute the grouped embedding microcheck under the training runner's card lock."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
STUDY = ROOT.parent
sys.path.insert(0, str(STUDY / "runner"))
import run_npu8 as runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    artifacts = STUDY.parent

    def mount(path):
        return str(Path("/artifacts") / path.resolve().relative_to(artifacts))

    with runner.LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        container, init = runner.inspect_container(artifacts, output)
        runner.check_card(output, "before", container, init)
        command = [*runner.DOCKER, "exec", "--env", "TASK_QUEUE_ENABLE=1",
                   "--env", "PYTHONDONTWRITEBYTECODE=1", container, "bash", "-c",
                   'source /workspace/env.sh && unset ASCEND_RT_VISIBLE_DEVICES && exec python3 -B "$@"',
                   "micro", mount(args.source / "micro_gate.py"), "--device", "npu", "--out",
                   mount(output / "micro.json")]
        (output / "command.json").write_text(json.dumps(command, indent=2))
        with (output / "run.log").open("wb") as logfile:
            status = subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT)
        runner.check_card(output, "after", container, init)
        status.check_returncode()


if __name__ == "__main__":
    main()
