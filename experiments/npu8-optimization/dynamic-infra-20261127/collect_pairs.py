"""Package completed pairs under reproducible labels for local analysis."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
HERE = Path(__file__).resolve().parent

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("recipe")
    ap.add_argument("--extra", action="append", default=[])
    args = ap.parse_args()
    dest = HERE / "receipts"
    dest.mkdir(exist_ok=True)
    reg = HERE / (args.recipe + "-runs.json")
    jobs = json.loads(reg.read_text()) if reg.is_file() else []
    for extra in args.extra:
        row = json.loads(Path(extra).read_text())
        row["mode"] = "l1-late"
        jobs.append(row)
    for row in jobs:
        if row["exit"]: continue
        label = "-".join([args.recipe,row["mode"],row["order"]])
        out = dest / (label + ".tgz")
        if out.exists(): continue
        temp = out.with_suffix(".partial")
        if temp.exists(): temp.unlink()
        subprocess.run([sys.executable, str(HERE / "reporting/collect_receipts.py"),
                        "--run", label + "=" + row["directory"], "--out", str(temp)], check=True)
        temp.rename(out)
if __name__ == "__main__":
    main()

