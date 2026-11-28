"""Keep original A/A training metadata, device samples, lock records and SPE recordings."""
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tarfile
import tempfile


HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from export import add_raw
from verify_slot import run as verify


def read(path):
    return json.loads(path.read_bytes())


def main():
    pair=HERE/"pair"
    result=read(pair/"result.json")
    assert result["exit"]==0
    raw=Path(result["directory"])
    rows=[]
    for line in (ROOT/"followup70-20261128/chip.jsonl").read_bytes().splitlines():
        row=json.loads(line)
        rows.append((datetime.fromisoformat(row["utc"]),row))
    windows=[]
    for leg in raw.glob("0?-?"):
        windows.append((datetime.fromisoformat(read(leg/"before.card.json")["utc"]),
                        datetime.fromisoformat(read(leg/"after.card.json")["utc"])+timedelta(seconds=1)))
    rows=[(t,v) for t,v in rows if any(a<=t<b for a,b in windows)]
    with tempfile.TemporaryDirectory(dir=HERE) as folder:
        temp=Path(folder)
        (temp/"pairs").mkdir()
        (temp/"pairs"/"stream-mutex-aa").symlink_to(pair,target_is_directory=True)
        slot=verify(temp,rows,None)
    assert all(not v["invalid"] and not v["unknown_or_external"] and v["snapshots_with_worker"]
               for legs in slot.values() for v in legs),slot
    (HERE/"slot.json").write_text(json.dumps(slot,indent=2)+"\n")
    (HERE/"chip.jsonl").write_text("".join(json.dumps(v)+"\n" for _,v in rows))
    with tarfile.open(HERE/"evidence.tar.gz","w:gz",compresslevel=2) as arc:
        arc.add(pair,arcname="pair")
        add_raw(arc,raw)
        arc.add(ROOT/"archived-runner/dynamic_shapes_npu8.json",arcname="schedule.json")
        for name in ("chip.jsonl","slot.json","pair.log","sample.log","all.log","analysis.json","stock-stacks.json","cache.txt"):
            arc.add(HERE/name,arcname=name)
        for leg in ("01-A","02-B"):
            for suffix in ("commands.json","worker.json","locks.data","locks.txt","spe.data","all.command.json","all.data","all.txt"):
                name=leg+"."+suffix
                arc.add(HERE/name,arcname=name)


if __name__=="__main__": main()
