"""Run the pinned branch's semantic gates followed by its predeclared timing windows."""
import json
from pathlib import Path
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
exp = here.parent
study = exp.parent
ready = exp / "build/ready.json"
while not ready.is_file():
    time.sleep(3)
build = json.loads(ready.read_bytes())
assert build["commit"] == "e3cda6399cb194c813530222478ea018f5aca5c9"


def run(source, *args):
    subprocess.run([sys.executable, "-B", str(source), *map(str, args)], check=True, cwd=study)


run(here / "configure.py")
run(here / "series.py", "--phase", "audit")
gates = exp / "semantics/probes"
if not (gates / "verified.json").exists():
    run(exp / "semantics/run_gates.py", "--package", build["inner_package"], "--sha256", build["sha256"],
        "--dest", gates)
run(exp / "semantics/verify.py", "--probes", gates, "--audit", study / "pairs/branch-v210-fixed-cache-audit-01-ab",
    "--sha256", build["sha256"], "--output", exp / "semantics/verified.json")
run(here / "series.py", "--phase", "performance")
(exp / "training.complete").write_text(build["commit"] + "\n")
