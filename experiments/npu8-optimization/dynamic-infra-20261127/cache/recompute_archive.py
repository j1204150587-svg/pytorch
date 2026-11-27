"""Recompute producer-local reuse from raw cache TSVs, cross-checking original native totals."""
import argparse
import csv
import re
import shutil
import tarfile
import tempfile
from pathlib import Path

from summarize_cache import main as summarize


def recompute(source, schedule, steps):
    allowed = re.compile(r"(01-A|02-B)/(?:markers-[0-9]+|lookup-[0-9]+-[0-9]+)\.tsv")
    with tempfile.TemporaryDirectory() as temporary:
        tmp = Path(temporary)
        with tarfile.open(source / "raw-lookup-markers.tgz") as archive:
            for member in archive:
                if not member.isfile() or not allowed.fullmatch(member.name):
                    raise ValueError(f"unexpected raw diagnostic archive member {member.name}")
                path = tmp / member.name
                path.parent.mkdir(exist_ok=True)
                with archive.extractfile(member) as inp, path.open("xb") as out:
                    shutil.copyfileobj(inp, out)
        for leg in tmp.iterdir():
            summarize(leg, schedule, steps)
            for file in ("steps.csv", "operators.csv"):
                original, revised = (p / file for p in (source / leg.name, leg))
                with original.open(newline="") as a, revised.open(newline="") as b:
                    for before, after in zip(csv.DictReader(a), csv.DictReader(b), strict=True):
                        for field in ("step", "op", "hit", "miss", "ineligible", "miss_zero_key"):
                            if before.get(field) != after.get(field):
                                raise ValueError(f"native counts changed in {leg.name}/{file}, {field}")
                target = source / "per-producer" / leg.name / file
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(revised, target)
            shutil.copyfile(leg / "producer-steps.csv", target.parent / "producer-steps.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    recompute(args.source, args.schedule, args.steps)
