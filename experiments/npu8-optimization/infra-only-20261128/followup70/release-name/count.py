"""Count exact shipping release callbacks in original eight-step A/A training ranks."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


HERE=Path(__file__).resolve().parent
PAIR=HERE/(sys.argv[1] if len(sys.argv)>1 else "pair")
OUT=HERE/(sys.argv[2] if len(sys.argv)>2 else "counts")
GROUP="hstu_npu8_release_name_20261128"
TRACE=Path("/sys/kernel/tracing/uprobe_events")
LIB="/artifacts/infra-only-20261128/stock-overlay/embedding/torch_npu/lib/libtorch_npu.so"
CONTAINER="recsys-gr-ds019-npu8"
def event(line):
    fd=os.open(TRACE,os.O_WRONLY|os.O_APPEND)
    try: os.write(fd,(line+"\n").encode())
    finally: os.close(fd)


def find(leg):
    dispatch=PAIR/leg/"dispatch.json"
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        if dispatch.exists():
            info=json.loads(dispatch.read_bytes())
            pids=subprocess.check_output(["docker","top",CONTAINER,"-eo","pid"],text=True).splitlines()[1:]
            for n in pids:
                proc=Path("/proc")/n.strip()
                try:
                    if (info["output"].encode() in (proc/"cmdline").read_bytes().split(b"\0") and
                            b"LOCAL_RANK=0" in (proc/"environ").read_bytes().split(b"\0")):
                        return proc
                except (FileNotFoundError,ProcessLookupError):
                    pass
        time.sleep(.1)
    raise TimeoutError(leg)


def main():
    OUT.mkdir(exist_ok=True)
    cid=subprocess.check_output(["docker","inspect","-f","{{.State.Pid}}",CONTAINER],text=True).strip()
    binary=Path("/proc")/cid/"root"/LIB.lstrip("/")
    assert subprocess.check_output(["sha256sum",binary],text=True).split()[0]=="4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"
    selected={"all":0x3c5d120,"opapi":0x3c5d19c,"compile":0x3c5d1d0}
    added=[]
    try:
        for label,offset in selected.items():
            event(f"p:{GROUP}/{label} {binary}:0x{offset:x}")
            added.append(label)
        for leg in (sys.argv[3:] or ("01-A","02-B")):
            proc=find(leg)
            output=OUT/(leg+".csv")
            cmd=["perf","stat","-x",",","-o",str(output),"-p",proc.name]
            for label in selected:cmd+=["-e",GROUP+":"+label]
            before=LIB in (proc/"maps").read_text()
            stderr=(OUT/(leg+".stderr")).open("w")
            perf=subprocess.Popen(cmd,stderr=stderr)
            deadline=time.monotonic()+1.5
            fds=[]
            while time.monotonic()<deadline:
                fds=[]
                for fd in (Path("/proc")/str(perf.pid)/"fd").glob("*"):
                    try:
                        if "perf_event" in os.readlink(fd): fds.append(fd)
                    except FileNotFoundError:pass
                if fds: break
                if perf.poll() is not None: break
                time.sleep(.005)
            after=LIB in (proc/"maps").read_text()
            assert perf.wait(timeout=240)==0,(leg,cmd)
            stderr.close()
            values={}
            for line in output.read_text().splitlines():
                parts=line.split(",")
                if len(parts)>3 and parts[2].startswith(GROUP+":"):
                    values[parts[2].split(":")[-1]]=int(parts[0])
            assert values["all"]>=values["opapi"]>70000 and values["all"]>=values["opapi"]+values["compile"],values
            report=dict(pid=int(proc.name),command=cmd,pta_mapped_before=before,
                        observed_perf_fd=bool(fds),pta_mapped_after_observing_perf_fd=after,counts=values)
            (OUT/(leg+".counts.json")).write_text(json.dumps(report,indent=2)+"\n")
    finally:
        for label in reversed(added):event(f"-:{GROUP}/{label}")


if __name__=="__main__": main()
