"""Verify all score-tile correctness failures against the original machine receipt."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile


def verify(archive, schedule):
    with tarfile.open(archive) as packed:
        entries={m.name:packed.extractfile(m).read() for m in packed if m.isfile()}
    def get(name):
        x,=[v for k,v in entries.items() if k.endswith("/"+name)]
        return x
    data=json.loads(get("micro.json"))
    assert int(get("train.exit"))==0 and data["exact"] is False and not data["windows"]
    for edge in ("before","after"):
        c=json.loads(get(edge+".card.json"))
        assert (c["logical_device"],c["board"],c["chip"],c["physical_id"])==(8,4,0,8)
        assert c["health"]=="OK" and not c["host_pids"] and c["orphan_context"] is None
    assert data["counts"]=={"table_builds":1,"jit":134,"prepared":2,"direct":66,"capped":88}
    for plan in data["plans"].values():
        assert not plan["witness"]["allocates_workspace"] and not plan["witness"]["allocates_sync_lock"]
    sequence=json.loads(schedule.read_bytes())
    assert len(sequence)==len(set(map(tuple,sequence)))==60
    expected=[(s,kind,i%2) for i,s in enumerate(sequence) for kind in
              (("finite","signed_zero","nonfinite") if i in (0,1,3) else ("finite",))]
    assert [(c["shape"],c["case"],c["stream"]) for c in data["cases"]]==expected
    groups={}
    for group, count in (("edges",66),("ordinary_edges",66),("score_edges",12)):
        selected=[c[group] for c in data["cases"] if group in c]
        assert len(selected)==count
        keys={"output","dQ","dK","dV"} if group!="score_edges" else {"scores","dScores"}
        assert all(set(r)==keys for r in selected)
        groups[group]={key:sum(r[key]["different_finite_values"]>0 for r in selected) for key in sorted(keys)}
        assert all(value==count for value in groups[group].values())
    return {"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "source_sha256":{k:hashlib.sha256(v).hexdigest() for k,v in sorted(entries.items())},
            "cases":len(expected),"input_cases":dict(Counter(c["case"] for c in data["cases"])),
            "differing_finite_cases":groups,"counts":data["counts"],"timing_windows":len(data["windows"])}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for field in ("archive","schedule","output"):
        cli.add_argument(field,type=Path)
    a=cli.parse_args()
    a.output.write_text(json.dumps(verify(a.archive,a.schedule),indent=2)+"\n",encoding="utf8")
