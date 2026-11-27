"""Summarize the guarded NPU flatten application counts in the eight-step diagnostic."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def summarize(path):
    info, *calls = (json.loads(row) for row in path.read_text().splitlines())
    if {c["step"] for c in calls} != set(range(8)) or not all(c["native_flatten_scopes"] > 0 for c in calls):
        raise ValueError("incomplete native flatten scope coverage")
    summary = defaultdict(Counter)
    for row in calls:
        if row["native_fast_path_scopes"] not in (0, 1):
            raise ValueError("invalid native marker count")
        slot = summary[row["caller"]]
        slot["calls"] += 1
        slot["fast_path_hits"] += row["native_fast_path_scopes"]
        slot["cat_scopes"] += row["cat_scopes"]
        slot["single_tensor_calls"] += int(row["tensors"] == 1)
        slot["shared_storage_violations"] += int(row["distinct_storages"] != 1)
        slot["adjacency_violations"] += int(bool(row["disqualifying_metadata"].get("offset_not_adjacent", 0)))
    if info["flat_switch"] != "1":
        raise ValueError("guard switch not enabled")
    return dict(info=info, steps=sorted({tuple((r["step"], *r["shape"])) for r in calls}),
                callers=dict(summary), total_calls=len(calls), hits=sum(r["native_fast_path_scopes"] for r in calls))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("probe", type=Path)
    args = parser.parse_args()
    print(json.dumps(summarize(args.probe), indent=2))
