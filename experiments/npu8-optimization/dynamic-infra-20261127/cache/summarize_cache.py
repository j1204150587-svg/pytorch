"""Summarize diagnostic Mode 1 lookups using training worker step boundaries."""
import argparse
import bisect
import csv
import heapq
import json
from collections import Counter, defaultdict
from pathlib import Path


def events(path):
    tid = int(path.stem.rsplit("-", 1)[1])
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.startswith("#"):
                continue
            stamp, op, result, key, _workspace = line.rstrip().split("\t")
            yield int(stamp), tid, op, result, int(key, 16)


def write_csv(path, data):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


COLUMNS = ("hit", "miss", "ineligible", "miss_zero_key", "first_hit", "first_miss",
           "within_step_hit", "within_step_miss", "earlier_step_hit", "earlier_step_miss")


def main(directory, schedule, expected_steps):
    marker, = directory.glob("markers-*.tsv")
    marks = [list(map(int, row.split())) for row in marker.read_text().splitlines()]
    actual = [row[2:4] for row in marks[:-1]]
    expected = json.loads(schedule.read_text())[:expected_steps]
    stamps = [row[0] for row in marks]
    if (actual != expected or len(expected) != expected_steps or
            [row[1] for row in marks] != list(range(expected_steps + 1)) or
            marks[-1][2:4] != [0, 0] or stamps != sorted(set(stamps))):
        raise ValueError("diagnostic step markers disagree with the selected schedule or end boundary")
    pid = marker.stem.split("-")[1]
    paths = list(directory.glob(f"lookup-{pid}-*.tsv"))
    if not paths:
        raise ValueError("no Mode 1 cache API records from the training worker")
    previous_step = {}
    positive_keys = set()
    positive_by_step = {}
    metrics = defaultdict(Counter)
    by_thread = defaultdict(Counter)
    totals = Counter()
    for stamp, tid, op, result, key in heapq.merge(*(events(path) for path in paths)):
        step = bisect.bisect_right(stamps, stamp) - 1
        if step == expected_steps:
            break
        if result not in {"hit", "miss", "ineligible"} or op == "UNKNOWN":
            raise ValueError(f"unrecognized cache event {op} {result}")
        group = metrics[step, op]
        group[result] += 1
        thread_counts = by_thread[step, tid]
        thread_counts[result] += 1
        totals[result] += 1
        if result == "ineligible":
            continue
        if not key:
            if result != "miss":
                raise ValueError("zero-key cache hit")
            group["miss_zero_key"] += 1
            thread_counts["miss_zero_key"] += 1
            continue
        key_id = (tid, op, key)
        earlier = previous_step.get(key_id)
        relation = ("first" if earlier is None else
                    "within_step" if earlier == step else "earlier_step")
        group[f"{relation}_{result}"] += 1
        thread_counts[f"{relation}_{result}"] += 1
        previous_step[key_id] = step
        if result == "hit":
            positive_keys.add(key_id)
        positive_by_step[step] = len(positive_keys)
    if not totals["hit"] + totals["miss"]:
        raise ValueError("no actual executor lookup was observed")
    details = []
    aggregates = defaultdict(Counter)
    for (step, op), counts in sorted(metrics.items()):
        aggregates[step].update(counts)
        details.append(dict(step=step, op=op, **{k: counts[k] for k in COLUMNS}))
    if any(step not in aggregates for step in range(expected_steps)):
        raise ValueError("one or more training steps have no observed cache API calls")
    per_step = []
    seen_shapes = set()
    for step, shape in enumerate(actual):
        row = aggregates[step]
        current = tuple(shape)
        per_step.append(dict(step=step, batch=current[0], seq=current[1],
                             shape_seen=int(current in seen_shapes),
                             rss_end_kib=marks[step+1][4], peak_rss_end_kib=marks[step+1][5],
                             distinct_keys_previously_returning_executor=positive_by_step.get(step),
                             **{k: row[k] for k in COLUMNS}))
        seen_shapes.add(current)
    write_csv(directory / "steps.csv", per_step)
    write_csv(directory / "operators.csv", details)
    write_csv(directory / "producer-steps.csv", [
        dict(step=s, tid=tid, main_thread=int(tid == int(pid)), **{k: vals[k] for k in COLUMNS})
        for (s, tid), vals in sorted(by_thread.items())])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("schedule", type=Path)
    parser.add_argument("--steps", type=int, default=60)
    args = parser.parse_args()
    main(args.directory, args.schedule, args.steps)
