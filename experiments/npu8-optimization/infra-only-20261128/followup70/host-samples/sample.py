"""Sample the existing training process from the host during a diagnostic repeat."""
import json
from pathlib import Path
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
artifacts = root.parents[2]
container = "recsys-gr-ds019-npu8"
init = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Pid}}", container], text=True).strip()
container_root = Path("/proc") / init / "root"
for leg in ("01-A", "02-B"):
    dispatch = root / "pair" / leg / "dispatch.json"
    deadline = time.monotonic() + 400
    while time.monotonic() < deadline:
        if dispatch.exists():
            info = json.loads(dispatch.read_bytes())
            output = info["output"]
            log = Path(info["session"]) / leg / "run.log"
            if log.exists() and "VariableFallbackKernel" in log.read_text(errors="replace"):
                break
        time.sleep(0.3)
    else:
        raise TimeoutError(leg)
    processes = subprocess.check_output(["docker", "top", container, "-eo", "pid"], text=True).splitlines()[1:]
    pids = []
    for item in processes:
        pid = int(item.strip())
        proc = Path("/proc") / str(pid)
        try:
            if output.encode() in (proc / "cmdline").read_bytes().split(b"\0") and b"LOCAL_RANK=0" in (proc / "environ").read_bytes().split(b"\0"):
                pids.append(pid)
        except FileNotFoundError:
            pass
    if len(pids) != 1:
        raise RuntimeError(f"expected one rank for {leg}, got {pids}")
    pid = pids[0]
    time.sleep(4)
    cmd = ["perf", "record", "-F", "199", "-e", "cpu-clock:u", "-g", "--call-graph", "fp", "-o", str(root / (leg + ".data")), "-p", str(pid), "--", "sleep", "8"]
    (root / (leg + ".cmd.json")).write_text(json.dumps(dict(command=cmd, pid=pid, session=info["session"], leg=leg), indent=2))
    subprocess.run(cmd, check=True)
    with (root / (leg + ".report")).open("w") as out:
        subprocess.run(["perf", "report", "--stdio", "-i", str(root / (leg + ".data")), "--symfs", str(container_root), "--no-children", "-g", "none", "--sort", "comm,dso,symbol", "--percent-limit", "0.25"], stdout=out, check=True)
