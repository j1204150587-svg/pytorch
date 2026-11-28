"""Run the original training entry point with a single native switch per pair."""
import json
from pathlib import Path
import re
import subprocess
import sys

here = Path(__file__).resolve().parent
study = here.parents[1]
prefix = "goal70-release-ring"
for mode, order in (("audit","ab"), ("l0","ab"), ("no-profiler","ab"),
                    ("no-profiler","ba"), ("l0","ba"), ("l1","ab")):
    out = study / "pairs" / f"{prefix}-{mode}-{order}"
    recipe = prefix + ("-audit" if mode == "audit" else "")
    if not (out / "result.json").is_file():
        cmd = [sys.executable, "-B", "runner/host_pair.py", "--runner", "archived-runner",
               "--recipe", f"recipes/{recipe}.json", "--out", str(out), "--mode", mode, "--order", order]
        print(cmd, flush=True)
        subprocess.run(cmd, cwd=study, check=True)
    result = json.loads((out / "result.json").read_bytes())
    assert result["exit"] == 0
    if mode == "audit":
        assert result["audit_passed"] and result["audited_steps"] == 8
        counts = {}
        for leg, enabled in (("01-A", "0"), ("02-B","1")):
            raw = (Path(result["directory"]) / leg / "run.log").read_text(errors="replace")
            rows = re.findall(r"NPU_RELEASE_RING enabled=(\d+) published=(\d+) consumed=(\d+)", raw)
            assert rows and all(e == enabled and p == c for e,p,c in rows), (leg, rows)
            maximum = max(int(p) for e,p,c in rows)
            assert (maximum == 0 if enabled == "0" else maximum > 70000), rows
            counts[leg] = rows
        (here / "counts.json").write_text(json.dumps(counts, indent=2))
    print("verified", mode, order, result["directory"], flush=True)
