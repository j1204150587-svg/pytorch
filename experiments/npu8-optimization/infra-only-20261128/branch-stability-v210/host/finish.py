"""Record the slot state and stop only the samplers owned by this completed series."""
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess

exp = Path(__file__).resolve().parents[1]
study = exp.parent
items = json.loads((exp / "schedule.json").read_bytes())
assert len(items) == 35 and (exp / "training.complete").read_text().strip() == "e3cda6399cb194c813530222478ea018f5aca5c9"
for item in items:
    pair = study / "pairs" / item["id"]
    assert (pair / "slot-verified.json").is_file() and json.loads((pair / "result.json").read_bytes())["exit"] == 0
(exp / "sampling.stop").touch()
pid = int((exp / "chip.pid").read_text())
cmdline = Path(f"/proc/{pid}/cmdline")
if cmdline.exists() and cmdline.stat().st_size == 0:
    original = cmdline.read_bytes().split(b"\0")
    if original and any(val.endswith(b"/chip_occupancy.py") for val in original) and os.fsencode(exp / "chip.jsonl") in original:
        os.kill(pid, signal.SIGTERM)


def run(*args):
    return subprocess.run(args, check=True, text=True, stdout=subprocess.PIPE, timeout=12).stdout.strip()


base = "recsys-gr-ds019-npu8"
paths = [Path("/run/lock") / (base + tail) for tail in ("-infra-study.lock", ".lock")]
with paths[0].open("r") as first, paths[1].open("r") as second:
    available = {}
    for f in (first, second):
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            available[f.name] = True
        except BlockingIOError:
            available[f.name] = False
    out = dict(local_time=datetime.now().astimezone().isoformat(), validated_pairs=len(items),
               locks_available=available, device=run("npu-smi", "info", "-t", "proc-mem", "-i", "4", "-c", "0"),
               health=run("npu-smi", "info", "-t", "health", "-i", "4", "-c", "0"),
               mapping=run("npu-smi", "info", "-m"),
               container_state=run("docker", "inspect", "-f", "{{.State.Status}} {{.State.Pid}}", base),
               container_processes=run("docker", "top", base, "-eo", "pid,ppid,comm"))
out["traces"] = {name: hashlib.sha256((exp / name).read_bytes()).hexdigest() for name in ("chip.jsonl", "host-load.jsonl")}
(exp / "final-slot.json").write_text(json.dumps(out, indent=2) + "\n")
print(out["local_time"], out["validated_pairs"], out["device"], out["locks_available"])
