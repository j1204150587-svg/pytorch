"""Validate the factory on NPU and run the audited original dynamic training pairs."""
import json
from pathlib import Path
import re
import subprocess
import sys

here = Path(__file__).resolve().parent
study = here.parents[1]
name = "goal70-empty-stride"
inner = "/artifacts/infra-only-20261128"
for toggle in ("0", "1"):
    path = here / ("micro-" + toggle)
    if not (path / "after.card.json").is_file():
        subprocess.run([sys.executable, "-B", "operator_probe.py", "--package", inner + "/stock-overlay/" + name,
                        "--env", "ACLNN_CACHE_LIMIT=65536", "--env", "NPU_DYNAMIC_REBASE_FORWARD=1",
                        "--env", "NPU_DYNAMIC_REBASE_INDICES=1", str(path),
                        inner + "/followup70-20261128/empty-restride/micro.py", "@OUT/test.json", toggle],
                       cwd=study, check=True)
    assert (path / "train.exit").read_text().strip() == "0"
assert json.loads((here / "micro-0/test.json").read_bytes()) == json.loads((here / "micro-1/test.json").read_bytes())
for mode, order in (("audit","ab"), ("no-profiler","ab"), ("l0","ab"), ("no-profiler","ba"), ("l0","ba"), ("l1","ab")):
    out = study / "pairs" / f"{name}-{mode}-{order}"
    recipe = name + ("-audit" if mode == "audit" else "")
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
        for leg in ("01-A", "02-B"):
            raw = (Path(result["directory"]) / leg / "run.log").read_text(errors="replace")
            rows = re.findall(r"NPU_EMPTY_STRIDE skipped=(\d+) kept=(\d+)",raw)
            assert rows and all((int(s) == 0 if leg == "01-A" else int(s) > 0) for s,k in rows)
            counts[leg] = rows
        (here / "counts.json").write_text(json.dumps(counts,indent=2))
    print("verified", mode, order, result["directory"], flush=True)
