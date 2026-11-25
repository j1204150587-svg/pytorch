"""Compare injected and natural HSTU training steps against the same 08 baseline."""

import json
import math
from pathlib import Path
import sys

import torch


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    a, b, output = map(Path, sys.argv[1:4])
    report_a, report_b = (read(path / "correctness.json") for path in (a, b))
    assert report_a["status"] == report_b["status"] == "passed"
    assert report_a["shard_order"] == report_b["shard_order"]
    expected = ("finite_below_clip", "nan", "posinf", "neginf", "finite_recovery_above_clip",
                "real_gradient_1", "real_gradient_2", "real_gradient_3")
    assert report_a["adjacent_gradient_partitions"] == report_b["adjacent_gradient_partitions"] == [True] * len(expected)
    assert [row["case"] for row in report_a["steps"]] == list(expected)
    assert read(a / "pta_wrapper.json") == read(b / "pta_wrapper.json")
    assert read(a / "hstu_graph.json")["counts"] == read(b / "hstu_graph.json")["counts"]
    details = []
    for index, (left, right) in enumerate(zip(report_a["steps"], report_b["steps"], strict=True)):
        assert left["case"] == right["case"]
        assert math.isclose(left["loss"], right["loss"], rel_tol=1e-6, abs_tol=1e-7)
        for key in ("shards", "before", "after", "overflow", "applied", "skip_delta", "clip", "cleared"):
            assert left[key] == right[key], (left["case"], key)
        grads = [torch.load(path / f"gradients-{index}.pt", map_location="cpu", weights_only=True) for path in (a, b)]
        for ga, gb in zip(*grads, strict=True):
            assert ga.shape == gb.shape and ga.dtype == gb.dtype
            assert torch.equal(ga.contiguous().view(-1).view(torch.uint8), gb.contiguous().view(-1).view(torch.uint8))
        details.append({"case": left["case"], "gradients": len(grads[0]), "applied": left["applied"]})
    output.write_text(json.dumps({"status": "passed", "comparison": "bitwise", "steps": details}, indent=2) + "\n")


if __name__ == "__main__":
    main()
