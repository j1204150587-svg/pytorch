"""Recompute completed A/B pairs directly from the runner's on-host registries."""
import json
from pathlib import Path
import subprocess
import sys

STUDY = Path(__file__).resolve().parents[1]
REPLICATES = {"meta-check": "meta-views", "cross-check": "cross-copy"}


def update(recipe_name):
    family = REPLICATES.get(recipe_name, recipe_name)
    config = json.loads((STUDY / (family + ".json")).read_text())
    a, b = config["A"], config["B"]
    changes = [k for k in a.keys() | b.keys() if a.get(k) != b.get(k)]
    if len(changes) != 1:
        raise ValueError(f"Expected one switch in {family}")
    output = STUDY / "reporting" / family
    args = [sys.executable, "-B", str(STUDY / "reporting/recompute_pairs.py"),
            "--switch", changes[0], "--schedule", str(STUDY / "runner/dynamic_shapes_npu8.json"),
            "--output", str(output)]
    for name in [family, *(item for item, base in REPLICATES.items() if base == family)]:
        registry = STUDY / (name + "-runs.json")
        if not registry.is_file():
            continue
        series = "r1" if name == family else name
        for entry in json.loads(registry.read_text()):
            if entry["exit"] != 0 or entry["order"] not in ("aa", "ab", "ba"):
                continue
            tag = f'{series}/{entry["order"]}/{entry["mode"]}'
            args.extend(["--pair", f'{tag}={entry["directory"]}'])
    if "--pair" in args:
        subprocess.run(args, check=True)


if __name__ == "__main__":
    update(sys.argv[1])
