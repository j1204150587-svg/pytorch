"""Download accepted runner receipts over verified SSH and recompute reports on the local machine."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from datetime import datetime, timedelta
import statistics
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time

from measure import COMPARISONS, digest, json_text, local_path, proof, read, slot_proof, write


LABELS={
    "fixed-cache":("同一分支包，三项关闭，65,536 项缓存","同一分支包，三项开启，65,536 项缓存"),
    "stock-to-full":("原镜像库、默认缓存","新分支包、三项开启、65,536 项缓存"),
    "default-cache":("同一分支包，三项关闭，默认缓存","同一分支包，三项开启，默认缓存")}
HERE=Path(__file__).resolve().parent


def unpack(archive,record,data):
    if digest(archive)!=record["sha256"] or archive.stat().st_size!=record["bytes"]:
        raise ValueError(f"packet checksum: {archive}")
    src="sources/"+record["id"]+".json"
    with tarfile.open(archive,"r:gz") as tar:
        all_members=tar.getmembers()
        if not all(m.isfile() and not PurePosixPath(m.name).is_absolute() and
                   all(x not in (".","..") and ":" not in x and "\\" not in x for x in PurePosixPath(m.name).parts)
                   for m in all_members):
            raise ValueError("unexpected archive entry")
        index=json.load(tar.extractfile(src))
        named={m.name for m in all_members}
        if len(named)!=len(all_members) or named!=set(index["members"])|{src}:
            raise ValueError("packet membership differs")
        for m in sorted(all_members,key=lambda m:m.name==src):
            payload=tar.extractfile(m).read()
            if m.name!=src:
                spec=index["members"][m.name]
                if len(payload)!=spec["bytes"] or hashlib.sha256(payload).hexdigest()!=spec["sha256"]:
                    raise ValueError(m.name)
            path=data/PurePosixPath(m.name)
            path.parent.mkdir(parents=True,exist_ok=True)
            if path.exists():
                if path.read_bytes()!=payload:
                    raise ValueError(f"immutable input changed: {path}")
                continue
            tmp=path.with_suffix(path.suffix+".part");tmp.write_bytes(payload);tmp.replace(path)


def index_data(data,out):
    schedule=read(data/"expected-schedule.json")
    occupancy={}
    inventory=[]
    audits={}
    ready=read(data/"build-ready.json")
    for comparison in COMPARISONS:
        rows=[]
        for item in schedule:
            if item["comparison"]!=comparison:continue
            name=item["id"]
            source=data/"sources"/(name+".json")
            if not source.is_file():continue
            receipt=read(source)
            raw_id=PurePosixPath(receipt["raw"]).name
            occ=slot_proof(data/"pairs"/name,data/"raw"/raw_id,data/"chips"/(name+".jsonl"),item["order"])
            load_file=data/"loads"/(name+".jsonl")
            samples=[json.loads(r) for r in load_file.read_bytes().splitlines()] if load_file.exists() else []
            for arm,v in occ.items():
                start=datetime.fromisoformat(v["begin"])
                stop=datetime.fromisoformat(v["end"])+timedelta(seconds=1)
                relevant=[x for x in samples if start<=datetime.fromisoformat(x["utc"])<stop]
                loads=[float(x["loadavg"].split()[0]) for x in relevant]
                v["host_load"]=dict(n=len(loads),min_1m=min(loads) if loads else None,
                                    median_1m=statistics.median(loads) if loads else None,max_1m=max(loads) if loads else None)
                if len(relevant)>1:
                    first,last=([int(t) for t in x["cpu_counts"][:8]] for x in (relevant[0],relevant[-1]))
                    dif=[b-a for a,b in zip(first,last)]
                    v["host_load"]["whole_machine_busy_pct"]=100*(sum(dif)-dif[3]-dif[4])/sum(dif) if sum(dif)>0 else None
            occupancy[name]=occ
            inventory.append({"id":name,"comparison":comparison,"mode":item["mode"],"order":item["order"],"raw":receipt["raw"],
                              "receipt":receipt["receipt"],"source_sha256":digest(source),"traces":receipt["profile_sources"],
                              "packet":read(data/"packets"/(name+".json"))})
            if item["mode"]=="audit":
                args=dict(receipt="pairs/"+name,chip_window="chips/"+name+".jsonl")
                evidence=proof(args,data,data/"raw"/raw_id,item["order"],"audit",comparison)
                result=read(data/"pairs"/name/"result.json")
                assert result["audit_passed"] and result["audited_steps"]==8
                for arm in "AB":
                    native=[v for path,v in evidence["legs"][arm]["library_digests"].items() if path.endswith("/libtorch_npu.so")]
                    assert len(native)==1 and (native[0]==ready["sha256"])!=(comparison=="stock-to-full" and arm=="A")
                audits[name]=dict(worker_proof=evidence,training=result)
                continue
            rows.append(dict(id=name,mode=item["mode"],order=item["order"],raw="../raw/"+raw_id,
                             receipt="../pairs/"+name,chip_window="../chips/"+name+".jsonl"))
        labels=LABELS[comparison]
        manifest=dict(comparison=comparison,baseline=labels[0],candidate=labels[1],schedule="../dynamic_shapes_npu8.json",
                      branch_commit=ready["commit"],branch_library_sha256=ready["sha256"],pairs=rows)
        path=data/"manifests"/(comparison+".json")
        path.parent.mkdir(exist_ok=True)
        body=json_text(manifest)
        existing=out/comparison/"metrics.json"
        if (path.is_file() and path.read_text(encoding="utf-8")==body and existing.is_file()
                and read(existing).get("analysis_sha256")==digest(HERE/"measure.py")):
            continue
        write(path,body)
        subprocess.run([sys.executable,"-B",str(HERE/"measure.py"),str(path),"--out",str(out/comparison)],check=True)
    write(data/"occupancy.json",json_text(occupancy))
    write(data/"audits.json",json_text(audits))
    write(data/"evidence-index.json",json_text(inventory))
    return len(inventory)


class Retrieval:
    def __init__(self,args,password):
        self.args=args
        self.password=password
        self.client=None
        self.remote_script=args.archive_dir+"/pack_receipts.py"

    def connect(self):
        import paramiko
        if self.client:self.client.close()
        client=paramiko.SSHClient()
        client.load_host_keys(str(self.args.known_hosts))
        client.connect(self.args.host,username=self.args.user,password=self.password,allow_agent=False,
                       look_for_keys=False,timeout=20,banner_timeout=25,auth_timeout=25)
        client.get_transport().set_keepalive(15)
        self.client=client
        self.command(["mkdir","-p",self.args.archive_dir])
        with self.client.open_sftp() as sftp:
            sftp.put(str(HERE/"pack_receipts.py"),self.remote_script)

    def command(self,argv):
        _,stdout,stderr=self.client.exec_command(shlex.join(argv),timeout=240)
        data=stdout.read().decode();error=stderr.read().decode()
        status=stdout.channel.recv_exit_status()
        if status:raise RuntimeError(f"host pack stage failed ({status}): {error[-1600:]}")
        return data

    def call(self,pair=None):
        argv=["nice","-n","19","ionice","-c","3","python3","-B",self.remote_script,
              "--study",self.args.study,"--archive-dir",self.args.archive_dir]
        if pair:argv+=["--pair",pair]
        return json.loads(self.command(argv))

    def fetch(self,record,path):
        if path.exists() and open_valid(path,record):return
        part=path.with_suffix(".part")
        if part.exists() and part.stat().st_size>record["bytes"]:
            raise ValueError(f"unexpected partial length: {part}")
        with self.client.open_sftp() as sftp:
            sftp.get_channel().settimeout(45)
            with sftp.open(record["archive"],"rb") as handle,part.open("ab") as dst:
                while dst.tell()<record["bytes"]:
                    count=min(1024*1024,record["bytes"]-dst.tell())
                    chunks=[(dst.tell()+i,min(32768,count-i)) for i in range(0,count,32768)]
                    for block in handle.readv(chunks,max_concurrent_prefetch_requests=8):
                        if not block:raise EOFError("short SFTP read")
                        dst.write(block)
        if not open_valid(part,record):raise ValueError(f"corrupt transfer {part}")
        part.replace(path)


def open_valid(path,record):
    return path.stat().st_size==record["bytes"] and digest(path)==record["sha256"]


def run(args):
    data=local_path(Path.cwd(),args.data)
    out=local_path(Path.cwd(),args.out)
    (data/"packets").mkdir(exist_ok=True,parents=True)
    out.mkdir(exist_ok=True,parents=True)
    if args.local_only:
        for receipt in sorted((data/"packets").glob("*.json")):
            unpack(receipt.with_suffix(".tgz"),read(receipt),data)
        print("local closed pairs",index_data(data,out));return
    import paramiko
    client=Retrieval(args,getpass.getpass("SSH password: "))
    previous=None
    errors=0
    try:
        while True:
            try:
                if not client.client or not client.client.get_transport().is_active():client.connect()
                state=client.call()
                current=len(state["accepted"]),len(state["schedule"]),state["build_ready"]
                if current!=previous:
                    print("accepted/scheduled/build",*current,flush=True);previous=current
                fresh=False
                for name in state["accepted"]:
                    if (data/"sources"/(name+".json")).is_file():continue
                    record=client.call(name)
                    path=data/"packets"/(name+".tgz")
                    client.fetch(record,path)
                    write(path.with_suffix(".json"),json_text(record))
                    unpack(path,record,data)
                    print("retrieved",name,record["bytes"],record["sha256"],flush=True)
                    fresh=True
                if fresh or ((data/"expected-schedule.json").exists() and not (data/"evidence-index.json").exists()):
                    index_data(data,out)
                if not args.watch or current[1]>0 and current[0]==current[1]:return
                errors=0
            except (OSError,EOFError,paramiko.SSHException) as e:
                errors+=1
                print("SSH retry",errors,type(e).__name__,flush=True)
                if client.client:client.client.close()
                client.client=None
                if errors>=8 or not args.watch:raise
            time.sleep(args.poll_seconds)
    finally:
        if client.client:client.client.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host",default="113.46.15.88")
    p.add_argument("--user",default="root")
    p.add_argument("--known-hosts",type=Path,default=HERE.parents[2]/"host88.known_hosts")
    p.add_argument("--study",default="/data/recsys-gr-ds019-host88/artifacts/infra-only-20261128")
    p.add_argument("--archive-dir",default="/mnt/paas/recsys-gr-ds019-host88-archive/dispatch-study-20261119-npu8/branch-stability-v210/receipts")
    p.add_argument("--data",type=Path,default=HERE.parent/"data")
    p.add_argument("--out",type=Path,default=HERE.parent/"reports")
    p.add_argument("--watch",action="store_true")
    p.add_argument("--local-only",action="store_true")
    p.add_argument("--poll-seconds",type=int,default=55)
    run(p.parse_args())

