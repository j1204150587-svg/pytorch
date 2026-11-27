"""Reproduce one native recipe's tables from the named raw evidence directories."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    recipes = {p.stem: data for p in STUDY.glob("*.json")
               if isinstance((data := json.loads(p.read_text(encoding="utf-8"))), dict)
               and {"gate", "A", "B"} <= data.keys()}
    cli.add_argument("recipe", choices=sorted(recipes))
    cli.add_argument("--evidence", type=Path)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--available", action="store_true", help="report completed pairs while a run is in progress")
    cli.add_argument("--steady-repeat", action="append", default=[], metavar="LABEL",
                     help="add a separately named replicate with no-profiler AA, AB and BA pairs")
    args = cli.parse_args()
    recipe = recipes[args.recipe]
    a, b = recipe.get("A", {}), recipe.get("B", {})
    switch, = (k for k in a.keys() | b.keys() if a.get(k) != b.get(k))
    base = args.evidence.resolve() if args.evidence else Path("evidence")
    schedule = base / "baseline-steady/workspace/configs/dynamic_shapes_npu8.json"
    out = args.output.resolve() if args.output else Path("reporting") / args.recipe
    argv = [sys.executable, "-B", str(HERE / "recompute_pairs.py"), "--switch", switch,
            "--schedule", str(schedule), "--output", str(out)]
    for mode in ("no-profiler", "l0", "l1", "l1-late"):
        for order in ("ab", "ba"):
            directory = base / f"{args.recipe}-{mode}-{order}"
            if (args.available or mode.startswith("l1")) and not (STUDY / directory / "status.json").is_file():
                continue
            argv.extend(["--pair", f"r1/{order}/{mode}={directory}"])
    repeats = args.steady_repeat or (["cross-check"] if args.recipe == "cross-copy" else [])
    for repeat in repeats:
        for order in ("aa", "ab", "ba"):
            directory = base / f"{repeat}-no-profiler-{order}"
            if (STUDY / directory / "status.json").is_file():
                argv.extend(["--pair", f"{repeat}/{order}/no-profiler={directory}"])
    subprocess.run(argv, check=True, cwd=STUDY)


if __name__ == "__main__":
    main()
