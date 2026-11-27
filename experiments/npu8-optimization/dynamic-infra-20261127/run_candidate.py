"""Gate native candidates, then collect paired unprofiled, Level0 and Level1 runs."""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
BASE = Path(os.environ.get("DYNAMIC_INFRA_WORKSPACE", HERE / "20261127T033119Z-baseline-l1-341x7gd3" / "workspace"))

def run(*argv):
    subprocess.run([sys.executable, "-u", *map(str, argv)], check=True)

def main():
    candidate = sys.argv[1]
    recipe = HERE / (candidate + ".json")
    data = json.loads(recipe.read_text())
    gate = HERE / data["gate"]
    if not (gate / "comparison.json").is_file():
        run(HERE / "semantic/run_semantic_pair.py", "--runner-dir", HERE / "runner",
            "--workspace", BASE, "--output", gate, "--common-env", json.dumps(data["common"]),
            "--env-a", json.dumps(data["A"]), "--env-b", json.dumps(data["B"]))
    if not json.loads((gate / "comparison.json").read_text())["passed"]:
        raise SystemExit("semantic gate failed")
    run(HERE / "native_suite.py", recipe, *sys.argv[2:])

if __name__ == "__main__":
    main()

