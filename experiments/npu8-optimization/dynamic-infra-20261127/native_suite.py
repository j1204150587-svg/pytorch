"""Run one native recipe, gating paired performance runs on its recorded semantic result."""
import argparse
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "runner"))
from run_suite import run_pair

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recipe", type=Path)
    parser.add_argument("pairs", nargs="+", help="mode/order")
    args = parser.parse_args()
    recipe = args.recipe.resolve()
    data = json.loads(recipe.read_text())
    os.environ["DYNAMIC_INFRA_RECIPE"] = str(recipe)
    if "gate" in data and not json.loads((HERE / data["gate"] / "comparison.json").read_text())["passed"]:
        raise SystemExit("semantic mismatch")
    result_file = HERE / (recipe.stem + "-runs.json")
    entries = json.loads(result_file.read_text()) if result_file.exists() else []
    for pair in args.pairs:
        mode, order = pair.split("/")
        if any(e["exit"] == 0 and e["mode"] == mode and e["order"] == order for e in entries):
            continue
        r = run_pair("NATIVE", mode, order)
        entries.append(r)
        result_file.write_text(json.dumps(entries, indent=2))
        if r["exit"]:
            raise SystemExit(r["exit"])
        if recipe.stem in ("meta-views", "meta-check"):
            from reporting.host_windows import update
            update(recipe.stem)

if __name__ == "__main__":
    main()

