"""Recompute eligible 08 candidate measurements from exported rank metrics and Level0 CSVs."""

from collections import defaultdict
import json
from pathlib import Path
from statistics import median
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from summarize_pairs import profile, steady_state, training


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_leg(leg, folder, mode):
    params = read(folder / "command.json")
    assert params["variant"] == leg["variant"]
    assert (folder / "train.exit").read_text().strip() == "0"
    for when in ("before", "after"):
        card = read(folder / f"{when}.card.json")
        assert card["logical_device"] == 8 and card["health"] == "OK"
        assert not card["host_pids"] and not card["orphan_context"]
    rank = folder / "training/rank000"
    summary, rows = training(rank / "metrics.json", "no-profiler" if mode == "steady" else "l0")
    values = (steady_state(summary, rows) if mode == "steady" else profile(folder / "step_trace_time.csv", "l0"))
    assert leg["values"] == values["values"] and leg["window"] == values["window"]
    return values["values"]


def delta(a, b):
    return {key: {"reference": a[key], "candidate": b[key], "saved": a[key] - b[key],
                  "improvement_pct": 100 * (a[key] - b[key]) / a[key] if a[key] else None}
            for key in a}


def summarize(folder, mode):
    result = defaultdict(list)
    for index, pair in enumerate(read(folder / "result.json"), 1):
        order = pair["order"]
        name = f"{index:02d}-{'-'.join(order)}"
        trial = {"pair": f"{folder.name}/{name}", "order": order, "status": pair["status"]}
        if pair["status"] == "matched":
            values = [load_leg(leg, folder / name / f"{pos}-{variant}", mode)
                      for pos, (leg, variant) in enumerate(zip(pair["legs"], order, strict=True), 1)]
            assert max(rate for leg in pair["legs"] for rate in leg["hbm_percent"]) - min(
                rate for leg in pair["legs"] for rate in leg["hbm_percent"]) <= 2
            if order[0] != "reference":
                values.reverse()
            trial["comparison"] = delta(*values)
        else:
            trial["declined"] = pair.get("error", "HBM usage drift outside same-pair range")
        key = next((name for name in order if name != "reference"), "reference")
        result[key].append(trial)
    return result


def main():
    root = Path(sys.argv[1])
    steady = summarize(root / sys.argv[2], "steady")
    profile_data = defaultdict(list)
    for directory in sys.argv[3:]:
        for variant, pairs in summarize(root / directory, "level0").items():
            profile_data[variant].extend(pairs)
    result = {"steady": steady, "level0": profile_data}
    medians = {}
    for variant, pairs in steady.items():
        selected = [p["comparison"]["ms_per_step"]["improvement_pct"] for p in pairs if p["status"] == "matched"]
        if selected:
            medians[variant] = median(selected)
    result["steady_step_improvement_pct_median"] = medians
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
