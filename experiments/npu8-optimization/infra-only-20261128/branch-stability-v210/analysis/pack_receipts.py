"""Archive only closed pairs using the host runner's immutable receipts. No device access."""
import argparse
from datetime import datetime, timedelta
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import tarfile
import tempfile


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def json_bytes(v):
    return (json.dumps(v,indent=2)+"\n").encode()


def scan(study):
    exp=study/"branch-stability-v210"
    schedule=read(exp/"schedule.json") if (exp/"schedule.json").is_file() else []
    accepted=[r["id"] for r in schedule if (study/"pairs"/r["id"]/"slot-verified.json").is_file()]
    return dict(schedule=schedule,accepted=accepted,build_ready=(exp/"build/ready.json").is_file())


def snapshot_bytes(path,windows):
    rows=[]
    with path.open("rb") as f:
        for line in f:
            if not line.endswith(b"\n"):
                continue
            stamp=datetime.fromisoformat(json.loads(line)["utc"])
            if any(a<=stamp<b for a,b in windows):
                rows.append(line)
    return b"".join(rows)


def pack(study, dest, pair_id):
    exp=study/"branch-stability-v210"
    state=scan(study)
    assert pair_id in state["accepted"] and re.fullmatch(r"branch-v210-[a-z0-9-]+",pair_id)
    item=next(r for r in state["schedule"] if r["id"]==pair_id)
    pair=study/"pairs"/pair_id
    result=read(pair/"result.json")
    assert result["exit"]==0 and result["order"]==item["order"]
    raw=Path(result["directory"])
    assert raw.parent.resolve()==(study.parent/"dynamic-shapes-20261127").resolve() and raw.name.startswith("20")
    dest.mkdir(parents=True,exist_ok=True)
    archive=dest/(pair_id+".tgz")
    marker=dest/(pair_id+".archive.json")
    if marker.is_file() and archive.is_file():
        return read(marker)
    files={}
    def add(path,name):
        assert path.is_file(),path
        files[name]=path
    def select(base,prefix,names):
        for name in names:
            p=base/name
            if p.is_file():add(p,prefix+"/"+name)
    for p in pair.rglob("*"):
        if p.is_file() and p.suffix in (".json",".log",".exit",".txt",".csv"):
            add(p,"pairs/"+pair_id+"/"+p.relative_to(pair).as_posix())
    raw_prefix="raw/"+raw.name
    select(raw,raw_prefix,("container.json","plan.json","workspace.sha256","evidence.sha256","status.json","runner.exit"))
    for conf in ("dynamic_shapes_npu8.json","model_many_tables.json","deepspeed_zero2_bf16_overflow_on.json"):
        add(raw/"workspace/configs"/conf,raw_prefix+"/workspace/configs/"+conf)
    add(study/"archived-runner/dynamic_shapes_npu8.json","dynamic_shapes_npu8.json")
    add(exp/"schedule.json","expected-schedule.json")
    add(exp/"build/ready.json","build-ready.json")
    source_traces=[]
    windows=[]
    for leg in sorted(raw.glob("0?-?")):
        pre=raw_prefix+"/"+leg.name
        for p in leg.iterdir():
            if not p.is_file():continue
            if p.name in ("trace_view.json.gz","step_trace_time.csv") or p.suffix in (".json",".log",".exit",".sha256"):
                if p.name=="trace_view.json.gz" and item["mode"]!="l1":continue
                add(p,pre+"/"+p.name)
        select(leg/"rank000",pre+"/rank000",("metrics.json","runtime.json","state_audit.json","overflow_scan.json","model_summary.json"))
        for p in (leg/"rank000").glob("*counter*"):
            if p.is_file():add(p,pre+"/rank000/"+p.name)
        ledger=leg/"evidence.sha256"
        for line in ledger.read_text().splitlines():
            sha,rel=line.split(maxsplit=1)
            if Path(rel).name in ("trace_view.json.gz","trace_view.json","kernel_details.csv","step_trace_time.csv"):
                source_traces.append(dict(leg=leg.name,source=str(leg/rel),resolved=str((leg/rel).resolve()),sha256=sha,
                                          bytes=(leg/rel).stat().st_size))
                if item["mode"]=="l1" and Path(rel).name=="kernel_details.csv":
                    add(leg/rel,pre+"/"+rel)
        before,after=(read(leg/(q+".card.json")) for q in ("before","after"))
        windows.append((datetime.fromisoformat(before["utc"]),datetime.fromisoformat(after["utc"])+timedelta(seconds=1)))
    assert len(windows)==2
    payloads={"chips/"+pair_id+".jsonl":snapshot_bytes(exp/"chip.jsonl",windows)}
    if (exp/"host-load.jsonl").is_file():
        payloads["loads/"+pair_id+".jsonl"]=snapshot_bytes(exp/"host-load.jsonl",windows)
    index=dict(id=pair_id,item=item,raw=str(raw),receipt=str(pair),profile_sources=source_traces,members={})
    def append(t,name,data,source):
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644
        t.addfile(info,io.BytesIO(data))
        index["members"][name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),source=source)
    with tempfile.NamedTemporaryFile(dir=dest,suffix=".tmp",delete=False) as fd:
        path=Path(fd.name)
    try:
        with tarfile.open(path,"w:gz",compresslevel=1) as tar:
            for name,p in sorted(files.items()):
                append(tar,name,p.read_bytes(),str(p))
            for name,body in payloads.items():
                append(tar,name,body,"completed leg timestamp windows")
            index_name="sources/"+pair_id+".json"
            body=json_bytes(index);info=tarfile.TarInfo(index_name);info.size=len(body)
            tar.addfile(info,io.BytesIO(body))
        path.replace(archive)
    finally:
        path.unlink(missing_ok=True)
    saved=dict(id=pair_id,archive=str(archive),bytes=archive.stat().st_size,sha256=digest(archive))
    tmp=marker.with_suffix(".tmp");tmp.write_bytes(json_bytes(saved));tmp.replace(marker)
    return saved


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study",type=Path,required=True)
    p.add_argument("--archive-dir",type=Path,required=True)
    p.add_argument("--pair")
    args=p.parse_args()
    print(json.dumps(pack(args.study,args.archive_dir,args.pair) if args.pair else scan(args.study)))

