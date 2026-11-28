"""Recompute v1 device and Host timing plus semantic checks from the original archive."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from tempfile import TemporaryDirectory
import tarfile

from profile_check import calc


def verify(archive_path, schedule, target):
    with tarfile.open(archive_path) as archive:
        entries = {k.name: archive.extractfile(k).read() for k in archive if k.isfile()}
    def member(suffix):
        [data] = [raw for name, raw in entries.items() if name.endswith(suffix)]
        return data
    micro = json.loads(member("/micro.json"))
    assert int(member("/train.exit")) == 0 and micro["status"] == "passed"
    sequence = json.loads(schedule.read_bytes())
    assert len(sequence) == len(set(map(tuple,sequence))) == 60 and micro["schedule"] == sequence
    for dtype in ("torch.int32","torch.int64"):
        cases = [v for v in micro["cases"] if v["dtype"] == dtype]
        assert len(cases) == 60 and [c["shape"] for c in cases] == sequence
        assert all(c["stream"] == ("default","alternate")[i%2] and set(c["routes"]) == set("ASP")
                   and all(not x["diff"] for x in c["routes"].values()) for i,c in enumerate(cases))
    assert micro["initial_counts"] == {"stock":120,"triton":120,"cold_triton":2,"cpp_entered":118,"cpp_returned":118}
    assert len(micro["fallbacks"]) == 4 and all(not x["diff"] and any(k.startswith("fallback_") for k in x["routes"])
                                                for x in micro["fallbacks"])
    assert micro["cross_stream_release"] == {}
    for boundary in ("before","after"):
        card = json.loads(member("/"+boundary+".card.json"))
        assert (card["logical_device"],card["board"],card["chip"],card["physical_id"]) == (8,4,0,8)
        assert card["health"] == "OK" and card["host_pids"] == [] and card["orphan_context"] is None
    with TemporaryDirectory() as folder:
        trace = Path(folder)/"trace.json"
        receipt = Path(folder)/"micro.json"
        trace.write_bytes(member("/trace_view.json"))
        receipt.write_bytes(member("/micro.json"))
        physical = calc(trace, receipt)
    assert physical["all_verified"] and len(physical["checks"]) == 12
    assert all((c["A_count"],c["S_count"],c["P_count"]) == (9,1,1) for c in physical["checks"])
    target.mkdir(exist_ok=True,parents=True)
    (target/"physical-real.json").write_text(json.dumps(physical,indent=2)+"\n", encoding="utf8")
    windows = []
    for win in micro["windows"]:
        legs = {}
        for route, leg in win["legs"].items():
            assert len(leg["host_us"]) == leg["calls"] == 48
            assert median(leg["host_us"]) == leg["p50_host_us"]
            assert leg["per_call_us"] == leg["wall_us"]/leg["calls"]
            assert leg["counts"] == {"A":{"stock":48},"S":{"triton":48},"P":{"cpp_entered":48,"cpp_returned":48}}[route]
            legs[route] = {field:leg[field] for field in ("per_call_us","p50_host_us","p95_host_us")}
        windows.append(dict(mode=win["mode"],order=win["order"],legs=legs))
    result = {"archive_sha256":hashlib.sha256(archive_path.read_bytes()).hexdigest(),
              "source_hashes":{k:hashlib.sha256(v).hexdigest() for k,v in sorted(entries.items())},
              "correctness_cases":len(micro["cases"]),"physical_checks":len(physical["checks"]),
              "windows":windows}
    (target/"verified-v1.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")


if __name__ == "__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","schedule","output"):
        cli.add_argument(name,type=Path)
    args=cli.parse_args()
    verify(args.archive,args.schedule,args.output)
