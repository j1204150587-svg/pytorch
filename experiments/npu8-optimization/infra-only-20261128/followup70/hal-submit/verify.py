"""Check the sensor, worker ownership, clock-linked records and raw Level1 windows independently."""
from collections import Counter
import csv
from datetime import datetime,timedelta
import gzip
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile

root=Path(__file__).resolve().parent
with gzip.open(root/"calls.csv.gz","rt",encoding="utf-8-sig",newline="") as fp:
    rows=list(csv.DictReader(fp))
with tarfile.open(root/"evidence.tgz") as tar:
    def blob(n):
        return tar.extractfile(n).read()
    def read(n):
        return json.loads(blob(n))
    receipt="pairs/goal70-hal-submit-v2-l1-ab"
    pair=read(receipt+"/result.json")
    assert pair["exit"]==0 and read("excluded-v1-result.json")["exit"]!=0
    rank=pair["directory"].rsplit("/",1)[1]
    assert blob("probe.cpp")== (root/"probe.cpp").read_bytes()
    sample=[json.loads(l) for l in blob("chip-snapshot.jsonl").splitlines()]
    report={"card":{},"steps":{},"sensor":{}}
    for leg in ("01-A","02-B"):
        worker=read(receipt+"/"+leg+"/worker.json")
        raw="raw/"+rank+"/"+leg+"/"
        before,after=[read(raw+name+".card.json") for name in ("before","after")]
        for card in (before,after):
            assert [card[k] for k in ("logical_device","board","chip","physical_id","health","host_pids")]==[8,4,0,8,"OK",[]]
        left=datetime.fromisoformat(before["utc"])
        right=datetime.fromisoformat(after["utc"])+timedelta(seconds=1)
        band=[s for s in sample if left<=datetime.fromisoformat(s["utc"])<right]
        assert band and all(s["valid"] for s in band)
        assert all(p["pid"]==worker["pid"] and p["owner"]=="training" for s in band for p in s["processes"])
        active=sum(bool(s["processes"]) for s in band)
        assert active
        steps=list(csv.DictReader(io.StringIO(blob(raw+"step_trace_time.csv").decode("utf-8-sig"))))
        assert [int(r["Step"]) for r in steps]==[4,5]
        report["card"][leg]=dict(samples=len(band),with_worker=active,pid=worker["pid"],official_level1_steps=steps)
    bpid=read(receipt+"/02-B/worker.json")["container_pid"]
    files=[m for m in tar if m.name.startswith("capture-v2/"+str(bpid)+"-") and m.name.endswith(".bin")]
    size=struct.calcsize("<QQIIIIIi")
    for f in files:
        cap,used,dropped,length=struct.unpack("<4Q",tar.extractfile(f).read(32))
        assert 0<used<=cap and dropped==0 and size==length
        report["sensor"][f.name]=dict(used=used,capacity=cap,dropped=dropped)
summary=json.loads((root/"result.json").read_bytes())
assert int(summary["profiler_pid"])==bpid
for n in ("4","5"):
    group=[r for r in rows if r["step"]==n]
    linked=[r for r in group if r.get("device_start_ns")]
    singles=[r for r in linked if r["gap_ns"]]
    assert all(int(r["sqes"])==1 and int(r["return_code"])==0 for r in group)
    dest=Counter((r["sq_id"],r["stream"]) for r in linked)
    assert all(len({s for q,s in dest if q==k})==1 for k,_ in dest)
    s=summary["steps"][n]
    assert s["all"]["calls"]==len(group) and len(linked)==s["unique_device_link"]
    assert len({(r["tid"],r["acl_start_ns"]) for r in linked})==len(linked)
    assert sum(int(r["hal_in_gap_ns"]) for r in singles)/1e6==s["nominal_hal_in_successor_gap_ms"]
    assert sum(int(r["after_hal_ns"]) for r in singles)/1e6==s["nominal_after_hal_in_successor_gap_ms"]
    report["steps"][n]=dict(calls=len(group),matched=len(linked),complete_gap=len(singles))
with (root/"evidence.tgz").open("rb") as fp:
    report["archive_sha256"]=hashlib.file_digest(fp,"sha256").hexdigest()
(root/"verification.json").write_text(json.dumps(report,indent=2))
