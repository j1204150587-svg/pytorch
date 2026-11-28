"""Keep original eight-step gates, exact release type counters, ownership and raw commands."""
from datetime import datetime,timedelta
import json
from pathlib import Path
import sys
import tarfile
import tempfile


here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root))
from export import add_raw
from verify_slot import run as verify


def read(p):return json.loads(p.read_bytes())


def main():
    pair=here/"pair2"
    counters=here/"counts2"
    entry=read(pair/"result.json")
    assert entry["exit"]==0 and entry["audit_passed"] and entry["audited_steps"]==8
    raw=Path(entry["directory"])
    reports={}
    bounds=[]
    for leg in ("01-A","02-B"):
        count=read(counters/(leg+".counts.json"))
        worker=read(pair/leg/"worker.json")
        assert count["pid"]==worker["pid"]
        reports[leg]=count
        a=read(raw/leg/"before.card.json")["utc"]
        b=read(raw/leg/"after.card.json")["utc"]
        bounds.append((datetime.fromisoformat(a),datetime.fromisoformat(b)+timedelta(seconds=1)))
    rows=[]
    for line in (root/"followup70-20261128/chip.jsonl").read_bytes().splitlines():
        v=json.loads(line)
        t=datetime.fromisoformat(v["utc"])
        if any(a<=t<b for a,b in bounds):rows.append((t,v))
    with tempfile.TemporaryDirectory(dir=here) as t:
        t=Path(t);(t/"pairs").mkdir();(t/"pairs/release-name-aa").symlink_to(pair,target_is_directory=True)
        slot=verify(t,rows,None)
    assert all(not v["invalid"] and not v["unknown_or_external"] and v["snapshots_with_worker"]
               for group in slot.values() for v in group)
    (here/"activation.json").write_text(json.dumps(dict(legs=reports,gate=entry,slot=slot),indent=2)+"\n")
    (here/"chip.jsonl").write_text("".join(json.dumps(v)+"\n" for t,v in rows))
    with tarfile.open(here/"evidence.tar.gz","w:gz",compresslevel=2) as arc:
        arc.add(pair,arcname="pair");add_raw(arc,raw)
        arc.add(root/"archived-runner/dynamic_shapes_npu8.json",arcname="schedule.json")
        arc.add(counters,arcname="counts2")
        for name in ("chip.jsonl","count2.log","pair2.log","activation.json","cpu-run.log","cpu-build.log","cpu.csv","count.log","pair.log"):
            arc.add(here/name,arcname=name)


if __name__=="__main__":main()
