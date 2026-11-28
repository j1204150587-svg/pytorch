"""Resolve the original CPU-clock perf samples using the ASLR mappings in each recording."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def analyze(data, root):
    mmaps = subprocess.check_output(["perf","script","-i",data,"--show-mmap-events","--show-task-events"],
                                     text=True,stderr=subprocess.DEVNULL)
    bases = {}
    for m in re.finditer(r"PERF_RECORD_MMAP2 [^:]+: \[(0x[\da-f]+)\((0x[\da-f]+)\) @ ([\da-fx]+) [^\]]+\]: r-xp (.+)",mmaps):
        name=m[4].strip()
        if name.endswith(("/libc.so.6","/libruntime_v100.so")):
            bases[name] = int(m[1],16),int(m[3],0)
    assert len(bases)==2
    raw=subprocess.check_output(["perf","script","-i",data,"--symfs",root,"-F","comm,pid,tid,ip,sym,dso"],
                                text=True,stderr=subprocess.DEVNULL)
    counts=Counter()
    parents=Counter()
    for group in raw.split("\n\n"):
        lines=group.splitlines()
        if len(lines)<2: continue
        counts["all"]+=1
        frames=[]
        for line in lines[1:]:
            match=re.match(r"\s*([0-9a-f]+) .*? \((.*)\)\s*$",line)
            if match and match[2] in bases:
                lo,off=bases[match[2]]
                frames.append((Path(match[2]).name,int(match[1],16)-lo+off))
            else: frames.append(None)
        if frames[0]==("libc.so.6",0x84ea4):
            counts["libc_0x84ea4_leaf"]+=1
            if lines[0].startswith("acl_thread "):
                counts["acl_thread_libc_0x84ea4_leaf"]+=1
                if len(frames)>1 and frames[1]:
                    name,ip=frames[1]
                    parents[f"{name}+{hex(ip)}"]+=1
    with data.open("rb") as f: sha=hashlib.file_digest(f,"sha256").hexdigest()
    return dict(sha256=sha,counts=counts,acl_thread_leaf_immediate_parents=parents)


if __name__=="__main__":
    src,out=map(Path,sys.argv[1:3])
    result={}
    for leg in ("01-A","02-B"):
        worker=json.loads((src/"pair"/leg/"worker.json").read_bytes())
        result[leg]=analyze(src/(leg+".data"),Path("/proc")/str(worker["init_pid"])/"root")
    out.write_text(json.dumps(result,indent=2)+"\n")
