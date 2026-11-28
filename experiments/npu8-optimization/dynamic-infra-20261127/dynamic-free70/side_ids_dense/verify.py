"""Verify the outputs, full synchronous windows and physical work of the dense side-ID probe."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import tarfile
from tempfile import TemporaryDirectory

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"side_ids"))
from profile_check import calc


def verify(archive_file, schedule_file, output):
    with tarfile.open(archive_file) as packed:
        files={m.name:packed.extractfile(m).read() for m in packed if m.isfile()}
    def member(suffix):
        raw,=[r for n,r in files.items() if n.endswith("/"+suffix)]
        return raw
    report=json.loads(member("micro.json"))
    assert int(member("train.exit"))==0 and report["status"]=="passed"
    sequence=json.loads(schedule_file.read_bytes())
    assert len(sequence)==len(set(map(tuple,sequence)))==60 and report["shapes"]==sequence
    for dtype in ("torch.int32","torch.int64"):
        cases=[c for c in report["cases"] if c["dtype"]==dtype]
        assert [r["shape"] for r in cases]==sequence
        for i,row in enumerate(cases):
            assert row["stream_index"]==i%2 and set(row["routes"])==set("ASP")
            assert all(not v["diff"] for v in row["routes"].values())
    assert report["correctness_counts"]=={"stock":120,"triton":120,"cold":2,"entered":118,"returned":118}
    assert not report["early_release_diff"] and len(report["fallbacks"])==4
    assert all(not r["diff"] and any(q.startswith("fallback_") for q in r["counts"]) for r in report["fallbacks"])
    for name in ("before.card.json","after.card.json"):
        c=json.loads(member(name))
        assert (c["logical_device"],c["board"],c["chip"],c["physical_id"])==(8,4,0,8)
        assert c["health"]=="OK" and c["host_pids"]==[] and c["orphan_context"] is None
    with TemporaryDirectory() as tmp:
        trace,receipt=Path(tmp)/"trace.json",Path(tmp)/"micro.json"
        trace.write_bytes(member("trace_view.json"))
        receipt.write_bytes(member("micro.json"))
        physical=calc(trace,receipt,"DenseSideIDs/","returned",2)
    out_windows=[]
    for window in report["windows"]:
        legs={}
        for name,leg in window["legs"].items():
            assert len(leg["host_us"])==leg["calls"]==48
            assert median(leg["host_us"])==leg["host_p50_us"]
            legs[name]={k:leg[k] for k in ("calls","per_call_us","host_p50_us","counts")}
        out_windows.append({"order":window["order"],"legs":legs})
    output.mkdir(parents=True,exist_ok=True)
    (output/"physical.json").write_text(json.dumps(physical,indent=2)+"\n",encoding="utf8")
    result={"archive_sha256":hashlib.sha256(archive_file.read_bytes()).hexdigest(),
            "source_files":{n:hashlib.sha256(r).hexdigest() for n,r in sorted(files.items())},
            "physical_verified":physical["all_verified"],"correctness_cases":len(report["cases"]),
            "measured_windows":out_windows}
    (output/"verified.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for param in ("archive","schedule","out"):
        cli.add_argument(param,type=Path)
    args=cli.parse_args()
    verify(args.archive,args.schedule,args.out)
