"""Record just the original L1 worker's two producer threads and a sparse clock anchor."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

HERE = Path(__file__).resolve().parent
CAP = Path("/mnt/paas/recsys-gr-ds019-host88-archive/dispatch-study-20261119-npu8/infra-only-followup70/analysis/producer-sched")
CONTAINER = "recsys-gr-ds019-npu8"
GROUP = "hstu_npu8_producer_sync_20261128"
LIB = "usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libacl_rt_impl.so"
SHA = "4595d4cf50f064cc9d6a97688eecf5e95ef92a4291911e26dd301091997c37ee"

def save(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")

def command(*args):
    return subprocess.check_output(args, text=True).strip()

def tracefs(line):
    fd=os.open("/sys/kernel/tracing/uprobe_events",os.O_WRONLY|os.O_APPEND)
    try: os.write(fd,(line+"\n").encode())
    finally: os.close(fd)

def find(leg):
    dispatch = HERE / "pair" / leg / "dispatch.json"
    deadline = time.monotonic()+3600
    while time.monotonic()<deadline:
        if dispatch.exists():
            info=json.loads(dispatch.read_bytes())
            for pid in command("docker","top",CONTAINER,"-eo","pid").splitlines()[1:]:
                p=Path("/proc")/pid.strip()
                try:
                    if (info["output"].encode() in (p/"cmdline").read_bytes().split(b"\0") and
                        b"LOCAL_RANK=0" in (p/"environ").read_bytes().split(b"\0")):
                        return p,info
                except (FileNotFoundError,ProcessLookupError): pass
        time.sleep(.1)
    raise TimeoutError(leg)

def tids(proc):
    until = time.monotonic()+120
    while time.monotonic()<until:
        found={}
        for p in (proc/"task").iterdir():
            try:
                name=(p/"comm").read_text().strip()
                if name=="pt_autograd_0" or p.name==proc.name:
                    row=next(s for s in (p/"status").read_text().splitlines() if s.startswith("NSpid:"))
                    found["main" if p.name==proc.name else "autograd"]=dict(host=int(p.name),ns=int(row.split()[-1]),comm=name)
            except (FileNotFoundError,ProcessLookupError): pass
        if len(found)==2:return found
        time.sleep(.02)
    raise TimeoutError("autograd thread")

def clocks(stop, out):
    with out.open("w") as fp:
        while not stop.is_set():
            measurements=[]
            for _ in range(6):
                a=time.time_ns()
                m=time.clock_gettime_ns(time.CLOCK_MONOTONIC)
                raw=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
                b=time.time_ns()
                measurements.append([a,m,raw,b])
            fp.write(json.dumps(min(measurements,key=lambda p:p[3]-p[0]))+"\n")
            stop.wait(.025)

def collect(leg,container_root):
    proc,info=find(leg)
    ids=tids(proc)
    out=CAP/leg
    out.mkdir(parents=True,exist_ok=False)
    target=",".join(str(row["host"]) for row in ids.values())
    head=["perf","record","--clockid","mono","--no-inherit","-m","512","-t",target]
    commands=[
        head+["--switch-events","-e","cpu-clock:u","-F","499","--call-graph","fp","-o",str(out/"sched.data"),"--","sleep","28"],
        head+["-e",GROUP+":enter","-e",GROUP+":leave","-o",str(out/"anchor.data"),"--","sleep","28"]]
    save(out/"receipt.json",dict(leg=leg,pid=int(proc.name),threads=ids,commands=commands,session=info["session"],
         offsets=(proc/"timens_offsets").read_text(),host_offsets=Path("/proc/self/timens_offsets").read_text(),lib=str(container_root/LIB),sha256=SHA))
    stop=threading.Event()
    clock=threading.Thread(target=clocks,args=(stop,out/"clock.jsonl"))
    clock.start()
    try:
        with (out/"sched.log").open("w") as a,(out/"anchor.log").open("w") as b:
            ps=[subprocess.Popen(c,stderr=fp) for c,fp in zip(commands,(a,b))]
            codes=[p.wait() for p in ps]
            save(out/"record-exits.json",codes)
            assert all(rc in (0,-15) for rc in codes),codes
    finally:
        stop.set();clock.join()
    decode(out,container_root)
    print(leg,ids,flush=True)

def decode(out,container_root):
    for name,opts in (("sched",["--show-switch-events","--fields","comm,pid,tid,time,event,ip,sym,dso"]),
                      ("anchor",["--fields","comm,pid,tid,time,event,trace"])):
        args=["perf","script","--ns","--no-inline","--symfs",str(container_root),"-i",str(out/(name+".data"))]+opts
        with (out/(name+".txt")).open("w") as fp,(out/(name+"-script.log")).open("w") as er:
            subprocess.run(args,stdout=fp,stderr=er,check=True)
        assert (out/(name+".txt")).stat().st_size>0

def main():
    container_root=Path("/proc")/command("docker","inspect","-f","{{.State.Pid}}",CONTAINER)/"root"
    lib=container_root/LIB
    assert hashlib.sha256(lib.read_bytes()).hexdigest()==SHA
    added=[]
    try:
        # Impl includes the internal profiled ACL runtime scope. It has a zero-VMA/offset difference here.
        for kind,name in (("p","enter"),("r","leave")):
            tracefs(f"{kind}:{GROUP}/{name} {lib}:0x412d0")
            added.append(name)
        with ThreadPoolExecutor(max_workers=2) as executor:
            work=[executor.submit(collect,leg,container_root) for leg in ("01-A","02-B")]
            for f in work:f.result()
    finally:
        for name in reversed(added): tracefs("-:"+GROUP+"/"+name)

if __name__=="__main__":main()

