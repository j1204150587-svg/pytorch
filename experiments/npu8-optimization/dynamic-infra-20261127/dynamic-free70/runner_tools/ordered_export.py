"""Snapshot completed ordered pairs, their registered gates and all decided attempts."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import sys
import tarfile
import tempfile

from run import FREE70, STUDY, check, read

sys.path.insert(0,str(STUDY/"reporting"))
from collect_receipts import files


def export(root, output, families):
    plan=read(root/"plan.json")
    runs=read(root/"runs.json") if (root/"runs.json").is_file() else []
    selected=[r for r in runs if r["family"] in families]
    assert all(r["exit"]==0 for r in selected)
    allowed=set(plan["families"])
    check(set(families)<=allowed,"unknown family")
    manifest={}
    def add_bytes(archive,name,data):
        check(name not in manifest,"duplicate evidence path: "+name)
        item=tarfile.TarInfo(name)
        item.size=len(data)
        archive.addfile(item,io.BytesIO(data))
        manifest[name]={"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}
    def add_path(archive,src,name):
        if not src.is_file():
            return
        check(name not in manifest,"duplicate evidence path: "+name)
        item=tarfile.TarInfo(name);item.size=src.stat().st_size
        class Reader:
            def __init__(self,stream):
                self.stream=stream;self.digest=hashlib.sha256()
            def read(self,size):
                data=self.stream.read(size);self.digest.update(data);return data
        with src.open("rb") as raw:
            reader=Reader(raw)
            archive.addfile(item,reader)
        manifest[name]={"bytes":item.size,"sha256":reader.digest.hexdigest()}
    output.parent.mkdir(exist_ok=True,parents=True)
    check(not output.exists(),"output exists; use a new timestamped snapshot")
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        pending=Path(tmp)/"snapshot.tgz"
        with tarfile.open(pending,"w:gz",compresslevel=1) as archive:
            add_path(archive,root/"plan.json","plan.json")
            add_bytes(archive,"runs.json",json.dumps(selected,indent=2).encode())
            for name in families:
                work=FREE70/"runs"/name
                for n in ("recipe.json","runner-sha256.json"):
                    add_path(archive,work/n,f"registries/{name}/{n}")
                gate=Path(plan["families"][name]["gate"]).parent
                for f in files(gate):
                    add_path(archive,f,f"gates/{name}/"+f.relative_to(gate).as_posix())
            for r in selected:
                pair=Path(r["directory"]).resolve(strict=True)
                check(pair.parent==(root/"evidence").resolve() and read(pair/"status.json")["exit"]==0,
                      "pair is not a completed entry from this campaign")
                for f in files(pair):
                    add_path(archive,f,"evidence/"+pair.name+"/"+f.relative_to(pair).as_posix())
            for j,job in enumerate(plan["jobs"],1):
                if job["family"] not in families:
                    continue
                pair=root/f"{j:02d}-{job['family']}-{job['mode']}-{job['order']}"
                for decision in sorted(pair.glob("??-?/try-*/decision.json")):
                    attempt=decision.parent
                    dest="attempts/"+attempt.relative_to(root).as_posix()
                    rec=read(decision)
                    for n in ("request.json","decision.json","runner.log","chip.jsonl","process.json"):
                        add_path(archive,attempt/n,dest+"/"+n)
                    if rec["verdict"]!="accepted":
                        original=Path(rec["directory"]).resolve(strict=True)
                        check(original.parent==FREE70.resolve(),"incorrect failure source")
                        for f in files(original):
                            add_path(archive,f,dest+"/original/"+f.relative_to(original).as_posix())
            add_bytes(archive,"file-sha256.json",json.dumps(manifest,indent=2).encode())
        pending.rename(output)
    print(output,len(selected),"completed pairs;",len(manifest)-1,"hashed files")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry",required=True)
    parser.add_argument("--families",nargs="+")
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    check(re.fullmatch(r"[\w-]+",args.registry),"invalid registry")
    root=FREE70/"ordered-resume"/args.registry
    export(root,args.output,args.families or list(read(root/"plan.json")["families"]))
