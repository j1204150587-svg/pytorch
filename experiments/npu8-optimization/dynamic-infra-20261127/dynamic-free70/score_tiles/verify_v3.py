"""Verify corrected BF16 score keys on the full eager attention and check all timing windows."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile


def verify(archive,schedule):
    with tarfile.open(archive) as packed:
        members={m.name:packed.extractfile(m).read() for m in packed if m.isfile()}
    raw,=[r for k,r in members.items() if k.endswith("/micro.json")]
    data=json.loads(raw)
    shapes=json.loads(schedule.read_bytes())
    assert len(shapes)==len(set(map(tuple,shapes)))==60
    expected=[(s,c,i%2) for i,s in enumerate(shapes)
              for c in (("finite","signed_zero","nonfinite") if i in (0,1,3) else ("finite",))]
    assert len(expected)==66 and [(r["shape"],r["case"],r["stream"]) for r in data["cases"]]==expected
    for i,row in enumerate(data["cases"]):
        for key in ("ordinary_edges","edges"):
            assert set(row[key])=={"output","dQ","dK","dV"}
            assert all(x["differing_bf16_words"]==0 for x in row[key].values())
    score_rows=[r for r in data["cases"] if "score_edges" in r]
    assert len(score_rows)==12 and all(set(r["score_edges"])=={"scores","dScores"} and
            all(x["differing_bf16_words"]==0 for x in r["score_edges"].values()) for r in score_rows)
    assert data["exact"] and data["counts"]=={"table_builds":1,"jit":134,"prepared":2,"direct":66,"capped":88}
    for binary in data["plans"].values():
        assert not binary["witness"]["allocates_workspace"] and not binary["witness"]["allocates_sync_lock"]
    assert [r["order"] for r in data["windows"]]==["NSP","PSN","NSP","PSN"]
    for r in data["windows"]:
        assert r["N"]["calls"]=={} and r["S"]["calls"]=={"jit":96} and r["P"]["calls"]=={"direct":48,"capped":48}
    return {"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "source_hashes":{k:hashlib.sha256(v).hexdigest() for k,v in sorted(members.items())},
            "full_cases":len(expected),"score_cases":len(score_rows),"micro_counts":data["counts"],"windows":data["windows"]}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("archive","schedule","output"):
        cli.add_argument(n,type=Path)
    a=cli.parse_args()
    a.output.write_text(json.dumps(verify(a.archive,a.schedule),indent=2)+"\n",encoding="utf8")
