"""Verify the 128-byte Addmm storage experiment from its original machine archive."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

from packed_profile import profile


def verify(archive_path, schedule_path, saved_profile):
    with tarfile.open(archive_path) as archive:
        members = [m for m in archive if m.isfile()]
        roots = {PurePosixPath(m.name).parts[0] for m in members}
        root, = roots
        files = {m.name.removeprefix(root + "/"): archive.extractfile(m).read() for m in members}
        assert len(files) == len(members)
    read = lambda name: json.loads(files[name])
    assert int(files["train.exit"]) == 0
    for boundary in ("before", "after"):
        state = read(boundary + ".card.json")
        assert (state["logical_device"], state["board"], state["chip"], state["physical_id"]) == (8, 4, 0, 8)
        assert state["health"] == "OK" and state["host_pids"] == [] and state["orphan_context"] is None
    micro = read("micro.json")
    expected = json.loads(schedule_path.read_bytes())
    assert len(expected) == len(set(map(tuple, expected))) == 60
    assert [c["shape"] for c in micro["cases"]] == expected
    assert all(c["differing_by_layer"] == [0] * 8 for c in micro["cases"])
    assert micro["exact"] and micro["data_ptrs_identical"]
    assert [x["order"] for x in micro["windows"]] == ["AB", "BA", "AB", "BA"]
    cold = []
    for order, legs in (("AB", ("cold-0-A.json", "cold-1-B.json")),
                        ("BA", ("cold-3-A.json", "cold-2-B.json"))):
        a, b = [read(leg)["us_per_eight_addmm"] for leg in legs]
        cold.append({"order": order, "A_us": a, "B_us": b, "time_reduction": 1 - b / a})
    originals = {}
    for name, raw in files.items():
        if name.endswith("/trace_view.json"):
            leg = name.split("-", 3)[2]
            originals[leg] = {"source": root + "/" + name, "sha256": hashlib.sha256(raw).hexdigest(),
                              "steps": profile(json.loads(raw))}
    assert set(originals) == {"A", "B"}
    originals["tasks_same_by_step"] = [a["tasks"] == b["tasks"] for a, b in
                                        zip(originals["A"]["steps"], originals["B"]["steps"], strict=True)]
    # JSON changes tuples in the reproduced profile into arrays.
    assert json.loads(json.dumps(originals)) == json.loads(saved_profile.read_bytes())
    streams = {}
    for leg, stream, count in (("A", 11, 8), ("B", 78, 3)):
        assert [row["shape"] for row in originals[leg]["steps"]] == [expected[4], expected[5]]
        for row in originals[leg]["steps"]:
            assert row["workspace_count"] == count and len(row["calls"]) == 8
            tasks = Counter(map(tuple, row["tasks"]))
            assert tasks == {(stream, "aclnnAddmm_CastAiCore_Cast"): 8,
                             (stream, "aclnnAddmm_MatMulCommon_MatMulV2"): 8}
        streams[leg] = stream
    return {"archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "source_files": len(files), "uncompressed_source_bytes": sum(map(len, files.values())),
            "source_hashes": {name: hashlib.sha256(files[name]).hexdigest() for name in sorted(files)},
            "shapes": len(expected), "exact_bf16_projection_outputs": 8 * len(expected),
            "fresh_process": cold, "shared_process": micro["windows"], "physical_streams": streams,
            "profile_hashes": {leg: originals[leg]["sha256"] for leg in ("A", "B")}}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ("archive", "schedule", "saved_profile", "output"):
        parser.add_argument(field, type=Path)
    args = parser.parse_args()
    receipt = verify(args.archive, args.schedule, args.saved_profile)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")
