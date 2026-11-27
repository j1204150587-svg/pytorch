"""Resume missing measured pairs after external occupation of logical device 8."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runner"))
import run_npu8 as slot


def idle():
    # Only read the mapping and process table. The actual runner takes the lock
    # and repeats the full device and container checks before and after each leg.
    mapping = slot.capture(["npu-smi", "info", "-m"], timeout=20)
    report = slot.capture(["npu-smi", "info"], timeout=20)
    if mapping.returncode or report.returncode:
        raise RuntimeError("cannot read logical device mapping or status")
    try:
        slot.idle_card(report.stdout, mapping.stdout, 8)
    except ValueError as error:
        if "occupied by host PIDs" in str(error) or "unnamed device context" in str(error):
            return False
        raise
    return True


def run(recipe, pair):
    last = None
    while True:
        with slot.LOCK.open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            available = idle()
        if available:
            result = subprocess.run([sys.executable, "-u", str(HERE / "native_suite.py"),
                                     str(HERE / (recipe + ".json")), pair], check=False)
            if result.returncode == 0:
                return
            registry = json.loads((HERE / (recipe + "-runs.json")).read_text())
            last_run = Path(registry[-1]["directory"])
            status = json.loads((last_run / "status.json").read_text())
            if result.returncode != 12 or not any(reason in str(status.get("detail", "")) for reason in
                    ("occupied by host PIDs", "device open by another process", "training container has other processes")):
                raise SystemExit(result.returncode)
        if last != available:
            print(slot.utc(), recipe, pair, "waiting for exclusive device 8", flush=True)
        last = available
        time.sleep(55)


if __name__ == "__main__":
    for name, pair in (arg.split(":", 1) for arg in sys.argv[1:]):
        run(name, pair)
