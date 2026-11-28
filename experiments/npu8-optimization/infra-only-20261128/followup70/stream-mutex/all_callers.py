"""Observe direct and wrapper users of each stream's two mutexes at glibc entry."""
import json
from pathlib import Path
import re
import subprocess
import time
from sample import HERE, CONTAINER, tracer, write


GROUP="hstu_npu8_stream_all_20261128"
SYMBOLS={"lock":0x83430,"unlock":0x84fe0}


def main():
    init=subprocess.check_output(["docker","inspect","-f","{{.State.Pid}}",CONTAINER],text=True).strip()
    elf=Path("/proc")/init/"root/lib/aarch64-linux-gnu/libc.so.6"
    syms=subprocess.check_output(["readelf","--dyn-syms","-W",elf],text=True)
    added=[]
    try:
        for name,offset in SYMBOLS.items():
            assert re.search(rf"\b0*{offset:x}\s+\d+ FUNC.* pthread_mutex_{name}@@GLIBC_2.17\b",syms)
            tracer(f"p:{GROUP}/{name} {elf}:0x{offset:x} addr=%x0:x64 word=+0(%x0):u32 caller=%x30:x64")
            added.append(name)
        for leg in ("01-A","02-B"):
            path=HERE/(leg+".locks.txt")
            deadline=time.monotonic()+3600
            objs=set()
            while time.monotonic()<deadline:
                if path.exists():
                    for cls,addr in re.findall(r"hstu_npu8_stream_20261128:([sr])_end:.*?addr=(0x[\da-f]+)",path.read_text()):
                        objs.add(int(addr,16)-(0x28 if cls=="s" else 0x58))
                    if objs: break
                time.sleep(.05)
            if not objs: raise TimeoutError(leg)
            stats=json.loads((HERE/(leg+".worker.json")).read_bytes())
            pid=stats["pid"]
            proc=Path("/proc")/str(pid)
            assert proc.is_dir()
            dispatch=json.loads((HERE/"pair"/leg/"dispatch.json").read_bytes())
            assert dispatch["output"].encode() in (proc/"cmdline").read_bytes().split(b"\0")
            addrs=sorted(obj+off for obj in objs for off in (0x28,0x58))
            filt=" || ".join(f"addr == {hex(x)}" for x in addrs)
            cmd=["perf","record","-m","128","-o",str(HERE/(leg+".all.data"))]
            for name in SYMBOLS: cmd+=["-e",GROUP+":"+name,"--filter",filt]
            cmd+=["-p",str(pid),"--","sleep","0.45"]
            write(HERE/(leg+".all.command.json"),cmd)
            subprocess.run(cmd,check=True)
            with (HERE/(leg+".all.txt")).open("w") as f:
                subprocess.run(["perf","script","--ns","-i",str(HERE/(leg+".all.data")),
                                "-F","trace:comm,pid,tid,cpu,time,event,trace"],stdout=f,check=True)
    finally:
        for name in reversed(added): tracer(f"-:{GROUP}/{name}")


if __name__=="__main__": main()
