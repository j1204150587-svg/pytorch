"""Run matched orders, retain failed attempts and verify chip occupancy before accepting a pair."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
experiment = here.parent
study = experiment.parent
names = ("fixed-cache", "stock-to-full", "default-cache")
prefix = "branch-v210-"


def stamp():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_bytes())


def write(path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def schedule():
    rounds = [(name, "audit", 1, "ab") for name in names]
    for n in range(6):
        ordered = names[n % 3:] + names[:n % 3]
        for name in ordered:
            order = ("ab", "ba")[(n + names.index(name)) % 2]
            rounds.append((name, "no-profiler", n + 1, order))
        if n in (0, 2, 3, 5):
            number = (0, 2, 3, 5).index(n)
            for name in reversed(ordered):
                order = ("ab", "ba")[(number + names.index(name)) % 2]
                rounds.append((name, "l0", number + 1, order))
    for name in ("fixed-cache", "default-cache"):
        rounds.append((name, "l1", 1, "ab"))
    return [dict(comparison=c, mode=m, repeat=n, order=o,
                 id=f"{prefix}{c}-{m}-{n:02d}-{o}") for c, m, n, o in rounds]


def verify_slot(pair):
    entry = read(pair / "result.json")
    assert entry["exit"] == 0
    raw = Path(entry["directory"])
    samples = [json.loads(s) for s in (experiment / "chip.jsonl").read_bytes().splitlines() if s]
    out = {}
    for leg in sorted(pair.glob("0?-?")):
        worker = read(leg / "worker.json")
        before, after = [read(raw / leg.name / (k + ".card.json")) for k in ("before", "after")]
        for card in (before, after):
            assert [card[k] for k in ("logical_device", "board", "chip", "physical_id", "health", "host_pids")] == [8,4,0,8,"OK",[]]
        start = datetime.fromisoformat(before["utc"])
        stop = datetime.fromisoformat(after["utc"]) + timedelta(seconds=1)
        window = [s for s in samples if start <= datetime.fromisoformat(s["utc"]) < stop]
        assert window and all(s["valid"] for s in window), "invalid slot sample"
        assert all(p["owner"] == "training" and p["pid"] == worker["pid"] for s in window for p in s["processes"]), "unrelated NPU process"
        assert any(s["processes"] for s in window), "no live worker sample"
        out[leg.name] = dict(worker=worker["pid"], samples=len(window),
                             with_worker=sum(bool(s["processes"]) for s in window), flags=worker["actual_flags"])
    assert len(out) == 2
    return dict(checked=stamp(), raw=str(raw), legs=out)


def run_pair(item):
    pair = study / "pairs" / item["id"]
    verified = pair / "slot-verified.json"
    if verified.exists():
        assert read(pair / "result.json")["exit"] == 0
        return
    kind = "audit" if item["mode"] == "audit" else "performance"
    recipe = experiment / "recipes" / f"{item['comparison']}-{kind}.json"
    if not (pair / "result.json").is_file():
        cmd = [sys.executable, "-B", str(here / "endpoint_pair.py"), "--runner", "archived-runner",
               "--recipe", str(recipe), "--out", str(pair), "--mode", item["mode"], "--order", item["order"]]
        subprocess.run(cmd, check=True, cwd=study)
    write(verified, verify_slot(pair))
    print(stamp(), "accepted", item["id"], read(verified)["raw"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("audit", "performance"), required=True)
    args = parser.parse_args()
    items = schedule()
    write(experiment / "schedule.json", items)
    for item in items:
        if (item["mode"] == "audit") != (args.phase == "audit"):
            continue
        run_pair(item)
        if item["mode"] in ("l0", "l1"):
            # Offload this run's original profiler tree only after both postflight checks.
            tool = study / "followup70-20261128/archive_profile.py"
            subprocess.run([sys.executable, "-B", str(tool), item["id"]], check=True)
        if shutil.disk_usage(study).free < 2 * 1024**3:
            raise RuntimeError("insufficient space for next profiler run")
    print(stamp(), "complete", args.phase, flush=True)


if __name__ == "__main__":
    main()
