"""Verify a pair's card occupancy and materialize its E2E, Level0 and Level1 report."""
import argparse
import csv
from datetime import datetime,timedelta
import hashlib
import io
import json
from pathlib import Path
import tarfile

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        while chunk:=stream.read(1024*1024):
            h.update(chunk)
    return h.hexdigest()

def write(root,name,obj):
    (root/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n",encoding="utf8")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--samples",type=Path,required=True)
    args=parser.parse_args()
    root=args.directory
    samples=[json.loads(line) for line in args.samples.read_bytes().splitlines() if line]
    with tarfile.open(root/"summary.tgz") as summary:
        metrics=json.load(summary.extractfile("metrics.json"))
        level=json.load(summary.extractfile("level1/pair.json"))
    with tarfile.open(root/"raw.tgz") as raw:
        def read(name):return json.load(raw.extractfile(name))
        pairs=sorted(m.name for m in raw.getmembers() if m.name.startswith("pairs/") and m.name.endswith("/result.json"))
        index=[]
        slots={}
        proof=[]
        for member in pairs:
            base=member.rsplit("/",1)[0]
            result=read(member)
            assert result["exit"]==0
            path=result["directory"]
            session=path.rsplit("/",1)[1]
            index.append(dict(name=base.split("/")[1],directory=path))
            for i,name in enumerate(result["order"].upper(),1):
                leg=f"0{i}-{name}"
                worker=read(base+"/"+leg+"/worker.json")
                prefix="raw/"+session+"/"+leg+"/"
                before,after=[read(prefix+s+".card.json") for s in ("before","after")]
                for card in (before,after):
                    assert [card[k] for k in ("logical_device","board","chip","physical_id","health","host_pids")]==[8,4,0,8,"OK",[]]
                start=datetime.fromisoformat(before["utc"])
                end=datetime.fromisoformat(after["utc"])+timedelta(seconds=1)
                window=[s for s in samples if start<=datetime.fromisoformat(s["utc"])<end]
                assert window and all(s["valid"] for s in window)
                assert all(p["owner"]=="training" and p["pid"]==worker["pid"] for s in window for p in s["processes"])
                n=sum(any(p["pid"]==worker["pid"] for p in s["processes"]) for s in window)
                assert n>0
                slots[base.split("/")[1]+"/"+leg]=dict(count=len(window),with_worker=n,worker=worker["pid"],
                                                      start=before["utc"],end=after["utc"])
                proof.extend(window)
                if result["mode"]=="l0":
                    rows=list(csv.DictReader(io.TextIOWrapper(raw.extractfile(prefix+"step_trace_time.csv"),encoding="utf-8-sig")))
                    assert [int(r["Step"]) for r in rows]==[4,5,6]
                    ratio=sum(float(r["Free"]) for r in rows)/sum(float(r["Stage"]) for r in rows)
                    report=next(r for r in metrics["rounds"] if r["order"]==result["order"])["level0"][name]
                    assert abs(ratio-report["free_ratio"])<1e-12
                if result["audit"]:
                    assert result["audit_passed"] and result["audited_steps"]==8
        windows={r["utc"]:r for r in proof}
        (root/"chip-windows.jsonl").write_text("".join(json.dumps(windows[t])+"\n" for t in sorted(windows)))
        with tarfile.open(root/"windows-and-commands.tgz","w:gz",compresslevel=6) as dest:
            for member in raw:
                if not member.isfile(): continue
                if member.name.endswith(("/metrics.json","/step_trace_time.csv","/worker.json","/before.card.json","/after.card.json",
                                        "/command.json","/result.json","/dispatch.json","/recipe.json","/workspace.sha256","/plan.json","/validated.json")):
                    dest.addfile(member,raw.extractfile(member))
    write(root,"metrics.json",metrics)
    write(root,"physical-comparison.json",dict(compare=level["compare"],steps={leg:{k:{
        "calls":v["all"]["calls"],"shape":v["shape"],"streams":v["physical"],"acl":v["acl_api"],
        "enqueue":v["all"]["latency"],"single_launch_us":v["exposure"]["single_acl_launch_us"],
        "after_ready_free_ms":v["exposure"]["already_queued_free_overlap_ms"]} for k,v in level[leg]["steps"].items()} for leg in "AB"}))
    write(root,"evidence-index.json",dict(runs=index,chip_windows=slots,files={n:sha(root/n) for n in
          ("raw.tgz","summary.tgz","windows-and-commands.tgz","repro.tgz","chip-windows.jsonl")}))
    print(len(pairs),"pairs;",len(slots),"legs;",min(v["count"] for v in slots.values()),"minimum samples per leg")

if __name__=="__main__": main()
