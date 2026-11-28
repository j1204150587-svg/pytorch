"""Export one complete set of original paired runs and the matching chip process samples."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


here=Path(__file__).resolve().parent
root=here.parents[1]
sys.path.insert(0,str(root))
from export import add_raw
from verify_slot import run as verify_slots


def read(p):
    return json.loads(p.read_bytes())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder",type=Path)
    parser.add_argument("prefix")
    parser.add_argument("--audit-only",action="store_true")
    args=parser.parse_args()
    dest=args.folder.resolve()
    suffixes=("audit-ab",) if args.audit_only else ("audit-ab","no-profiler-ab","no-profiler-ba","l0-ab","l0-ba","l1-ab")
    pairs=[root/"pairs"/(args.prefix+"-"+suffix) for suffix in suffixes]
    for pair in pairs:
        assert read(pair/"result.json")["exit"]==0
    py=sys.executable
    if not args.audit_only:
        with (dest/"paired-endpoints.json").open("w") as out:
            subprocess.run([py,"-B",str(root/"runner/paired_metrics.py"),"--round",str(pairs[1]),str(pairs[3]),
                            "--round",str(pairs[2]),str(pairs[4]),"--schedule",str(root/"archived-runner/dynamic_shapes_npu8.json")],
                           stdout=out,check=True)
    rows=[json.loads(line) for line in (root/"followup70-20261128/chip.jsonl").read_bytes().splitlines() if line]
    windows=[]
    for p in pairs:
        legroot=Path(read(p/"result.json")["directory"])
        for leg in legroot.glob("0?-?"):
            windows.append([datetime.fromisoformat(read(leg/"before.card.json")["utc"]),
                            datetime.fromisoformat(read(leg/"after.card.json")["utc"])+timedelta(seconds=1)])
    filtered=[(datetime.fromisoformat(x["utc"]),x) for x in rows
              if any(a<=datetime.fromisoformat(x["utc"])<b for a,b in windows)]
    with tempfile.TemporaryDirectory(dir=dest) as tmp:
        tmp=Path(tmp)
        (tmp/"pairs").mkdir()
        for pair in pairs: (tmp/"pairs"/pair.name).symlink_to(pair,target_is_directory=True)
        check=verify_slots(tmp,filtered,None)
    assert all(not s["invalid"] and not s["unknown_or_external"] and s["snapshots_with_worker"]
               for legs in check.values() for s in legs),check
    (dest/"slot.json").write_text(json.dumps(check,indent=2)+"\n")
    with tarfile.open(dest/"evidence.tar.gz","w:gz",compresslevel=2) as archive:
        for p in pairs:
            archive.add(p,arcname="pairs/"+p.name)
            add_raw(archive,Path(read(p/"result.json")["directory"]))
        names=["audit.json","slot.json"]
        if not args.audit_only: names.append("paired-endpoints.json")
        for name in names:
            archive.add(dest/name,arcname=name)
        for sub in ("micro-0","micro-1"):
            for f in (dest/sub).glob("*"):
                if f.is_file() and f.name in {"before.card.json","after.card.json","micro.json","command.json",
                                             "train.exit","run.log","container.json"}:
                    archive.add(f,arcname=sub+"/"+f.name)
        archive.add(root/"archived-runner/dynamic_shapes_npu8.json",arcname="schedule.json")
        entries="".join(json.dumps(v)+"\n" for _,v in filtered)
        with tempfile.NamedTemporaryFile("w",encoding="utf-8",dir=dest) as t:
            t.write(entries)
            t.flush()
            archive.add(t.name,arcname="chip.jsonl")
    print(dest/"evidence.tar.gz",flush=True)


if __name__=="__main__": main()
