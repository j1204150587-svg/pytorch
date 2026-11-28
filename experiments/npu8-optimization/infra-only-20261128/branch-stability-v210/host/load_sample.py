"""Record shared-host CPU load alongside the independent chip-occupancy sampler."""
from datetime import datetime, timezone
import json
from pathlib import Path
import time

root = Path(__file__).resolve().parents[1]
with (root / "host-load.jsonl").open("x", buffering=1) as out:
    while not (root / "training.complete").exists() and not (root / "sampling.stop").exists():
        row = {"utc": datetime.now(timezone.utc).isoformat(),
               "loadavg": Path("/proc/loadavg").read_text().strip(),
               "cpu_counts": Path("/proc/stat").read_text().splitlines()[0].split()[1:]}
        pressure = Path("/proc/pressure/cpu")
        if pressure.exists():
            row["cpu_psi"] = pressure.read_text().strip().splitlines()
        print(json.dumps(row), file=out)
        time.sleep(6)
