"""Schedule the independent copy micro probes, semantic pair and original performance pairs."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
INNER = "/artifacts/infra-only-20261128"


def command(argv):
    print(time.strftime("%FT%T%z"), " ".join(map(str, argv)), flush=True)
    subprocess.run(list(map(str, argv)), cwd=ROOT, check=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual128",action="store_true")
    args=parser.parse_args()
    name="copy-rebase-128" if args.residual128 else "copy-rebase"
    root_pair = ROOT / "followup70-20261128/host-samples/pair/result.json"
    while not root_pair.exists():
        time.sleep(10)
    runs = HERE / ("runs-128" if args.residual128 else "runs")
    runs.mkdir(exist_ok=True)
    for toggle in ("0", "1"):
        out = runs / ("micro-" + toggle)
        if not (out / "micro.json").exists() or not (out / "after.card.json").exists():
            command([sys.executable, "-B", "operator_probe.py", "--package", INNER + "/stock-overlay/" + name,
                     "--env", "ACLNN_CACHE_LIMIT=65536", "--env", "NPU_DYNAMIC_REBASE_INDICES=1",
                     "--env", "NPU_DYNAMIC_REBASE_FORWARD=1", out, INNER + "/copy_rebase/" + ("micro128.py" if args.residual128 else "micro.py"),
                     "@OUT/micro.json", "--switch", toggle])
        summary = json.loads((out / "micro.json").read_bytes())
        assert summary["switch"] == toggle and summary["overlap_error_preserved"] and len(summary["checks"]) == (37 if args.residual128 else 25)
    for mode, order in (("audit", "ab"), ("no-profiler", "ab"), ("no-profiler", "ba"),
                        ("l0", "ab"), ("l0", "ba"), ("l1", "ab")):
        out = ROOT / "pairs" / f"{name}-{mode}-{order}"
        file = out / "result.json"
        if not file.exists():
            recipe = name + ("-audit" if mode == "audit" else "")
            command([sys.executable, "-B", "runner/host_pair.py", "--runner", "archived-runner",
                     "--recipe", f"recipes/{recipe}.json", "--out", out, "--mode", mode, "--order", order])
        entry = json.loads(file.read_bytes())
        assert entry["exit"] == 0
        if mode == "audit":
            assert entry["audit_passed"] and entry["audited_steps"] == 8
            counts = entry["native_counters"]
            symbol = "npu_inplace_copy_rebase_counter"
            a,b = counts["A"][symbol], counts["B"][symbol]
            assert not a["accepted"] and not a["keys"] and b["accepted"] and b["keys"] == b["accepted"]*2
            assert b["descriptors"] <= b["keys"]
            if args.residual128:
                assert b["accepted"] == 16384 and b["descriptors"] < b["keys"]/2
            for v in ("A","B"):
                for direction in ("forward","backward"):
                    assert counts[v][f"npu_embedding_{direction}_rebase_counter"]["accepted"] == 8192
        print("VERIFIED", out, flush=True)


if __name__ == "__main__":
    main()
