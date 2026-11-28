"""Collect the original audit, independent gates and their per-rank activation receipts."""
import argparse
from datetime import datetime,timedelta
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from verify import read

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
sys.path.insert(0,str(STUDY))
from export import add_raw

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probes",type=Path,required=True)
    ap.add_argument("--audit",type=Path,required=True)
    ap.add_argument("--sha256",required=True)
    args=ap.parse_args()
    verified=HERE/"activation.json"
    subprocess.run([sys.executable,"-B",str(HERE/"verify.py"),"--probes",str(args.probes),"--audit",str(args.audit),
                    "--sha256",args.sha256,"--output",str(verified)],check=True)
    raw=Path(read(args.audit/"result.json")["directory"])
    ranges=[]
    for leg in [*raw.glob("0?-?"),args.probes/"A",args.probes/"B"]:
        ranges.append((datetime.fromisoformat(read(leg/"before.card.json")["utc"]),
                       datetime.fromisoformat(read(leg/"after.card.json")["utc"])+timedelta(seconds=1)))
    journal=HERE.parent/"chip.jsonl"
    filtered=[]
    if journal.is_file():
        for line in journal.read_bytes().splitlines():
            if line and any(a<=datetime.fromisoformat(json.loads(line)["utc"])<b for a,b in ranges):
                filtered.append(line)
    (HERE/"matched-chip.jsonl").write_bytes(b"\n".join(filtered)+b"\n")
    with tarfile.open(HERE/"evidence.tar.gz","w:gz",compresslevel=3) as arc:
        arc.add(args.probes,arcname="probes")
        arc.add(args.audit,arcname="fixed-cache-audit")
        add_raw(arc,raw)
        for f in ("activation.json","matched-chip.jsonl"):
            arc.add(HERE/f,arcname=f)
        for f in (STUDY/"embedding/micro.py",STUDY/"followup70-20261128/overflow/check.py",
                  STUDY/"operator_probe.py",STUDY/"runner/reference.json",STUDY/"archived-runner/dynamic_shapes_npu8.json",
                  STUDY/"counter_dump.c",HERE.parent/"build/ready.json"):
            arc.add(f,arcname="source/"+f.relative_to(STUDY).as_posix())
    print(HERE/"evidence.tar.gz")

if __name__=="__main__":main()

