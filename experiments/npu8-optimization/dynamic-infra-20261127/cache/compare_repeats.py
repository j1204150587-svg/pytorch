"""Compare the same eight actual HSTU shapes on their first and second visits."""
import argparse
import csv
from collections import Counter
from pathlib import Path


def compare(directory):
    with (directory / "steps.csv").open(newline="") as stream:
        steps = list(csv.DictReader(stream))
    shapes = [(int(r["batch"]), int(r["seq"])) for r in steps]
    if len(shapes) != 16 or len(set(shapes[:8])) != 8 or shapes[:8] != shapes[8:]:
        raise ValueError("expected 8 distinct shapes followed by the same sequence")
    if [int(r["shape_seen"]) for r in steps] != [0] * 8 + [1] * 8:
        raise ValueError("shape seen flags disagree with repeated schedule")
    with (directory / "operators.csv").open(newline="") as stream:
        ops = {(int(r["step"]), r["op"]): r for r in csv.DictReader(stream)}
    rows = []
    sums = {}
    for idx, (batch, seq) in enumerate(shapes[:8]):
        for op in ("ALL", "aclnnEmbedding", "aclnnEmbeddingDenseBackward", "aclnnMean", "aclnnAdd", "aclnnDivs"):
            a, b = (steps[idx+j] if op == "ALL" else ops[idx+j, op] for j in (0, 8))
            values = {}
            for counter, slot in ((a, "first"), (b, "repeat")):
                for key in ("hit", "miss", "earlier_step_hit", "within_step_hit", "first_miss", "ineligible"):
                    values[f"{slot}_{key}"] = int(counter[key])
            rows.append(dict(shape_index=idx, batch=batch, seq=seq, op=op, **values))
            sums.setdefault(op, Counter()).update(values)
    for name, entries in (("repeat-shape-pairs.csv", rows),
                          ("repeat-op-totals.csv", [dict(op=op, **vals) for op, vals in sums.items()])):
        with (directory / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=entries[0])
            writer.writeheader()
            writer.writerows(entries)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg", type=Path)
    compare(parser.parse_args().leg)
