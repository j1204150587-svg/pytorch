"""Join raw chip process samples with original-runner boundaries and the observed worker."""
import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path

def read(p):
    return json.loads(p.read_bytes())

def run(root, samples, raw_root):
    report={}
    for pair in sorted((root/"pairs").iterdir()):
        if not (pair/"result.json").is_file(): continue
        entry=read(pair/"result.json")
        if entry["exit"]: continue
        raw=Path(entry["directory"])
        if raw_root: raw=raw_root/raw.name
        legs=[]
        for folder in sorted(pair.glob("0?-?")):
            leg=raw/folder.name
            before,after=(read(leg/(name+".card.json")) for name in ("before","after"))
            for marker in (before,after):
                assert (marker["logical_device"],marker["board"],marker["chip"],marker["physical_id"])==(8,4,0,8)
                assert marker["health"]=="OK" and not marker["host_pids"]
            begin=datetime.fromisoformat(before["utc"])
            end=datetime.fromisoformat(after["utc"])+timedelta(seconds=1)
            window=[x for t,x in samples if begin<=t<end]
            expected=read(folder/"worker.json")["pid"]
            foreign=[x for x in window if any(p["owner"]!="training" or p["pid"]!=expected for p in x["processes"])]
            legs.append(dict(leg=folder.name,worker=expected,samples=len(window),
                snapshots_with_worker=sum(any(p["pid"]==expected for p in x["processes"]) for x in window),
                invalid=sum(not x["valid"] for x in window),unknown_or_external=foreign))
        report[pair.name]=legs
    return report

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("root",type=Path)
    ap.add_argument("--raw-root",type=Path)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    samples=[(datetime.fromisoformat(x["utc"]),x) for line in (args.root/"chip.jsonl").read_bytes().splitlines() if line for x in [json.loads(line)]]
    report=run(args.root,samples,args.raw_root)
    args.out.write_text(json.dumps(report,indent=2))
    flagged={k:v for k,v in report.items() if any(r["unknown_or_external"] or r["invalid"] for r in v)}
    print(json.dumps({"pairs":len(report),"flagged":flagged}))

