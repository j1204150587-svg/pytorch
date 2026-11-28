"""Attribute complete original CPU-clock samples to the release and producer copy functions."""
from collections import Counter
import json
from pathlib import Path
import re
import subprocess
import sys


def run(study,leg):
    w=json.loads((study/"pair"/leg/"worker.json").read_bytes())
    lib,mapping=next(iter(w["libraries"].items()))
    base=int(mapping.split("-")[0],16)-int(mapping.split()[2],16)
    root=Path("/proc")/str(w["init_pid"])/"root"
    binary=root/lib.lstrip("/")
    symbols=subprocess.check_output(["nm","-S","-C","--defined-only",binary],text=True).splitlines()
    ranges=[]
    for row in symbols:
        m=re.match(r"([0-9a-f]+) ([0-9a-f]+) [a-zA-Z] (.*)",row)
        if m and (any(x in m[3] for x in ("::CopyReleaseParamFunc(","::ReleaseParamFunc(","::CopyFunc(",
                "::WriteToReleaseQueue(","::ReadFromReleaseQueue(","ExecuteParasOpApi::Copy(","ExecuteParasOpApi::Release("))):
            lo,sz=int(m[1],16),int(m[2],16)
            ranges.append((lo,lo+sz,m[3]))
    data=subprocess.check_output(["perf","script","-i",study/(leg+".data"),"--symfs",root,"-F","comm,pid,tid,ip,sym,dso"],
                                   text=True,stderr=subprocess.DEVNULL)
    counts=Counter()
    samples=[]
    for stack in data.split("\n\n"):
        lines=stack.splitlines()
        if len(lines)<2:continue
        counts["all"]+=1
        targeted=[]
        for i,line in enumerate(lines[1:]):
            if not line.rstrip().endswith("("+lib+")"): continue
            ip=int(line.split()[0],16)-base
            for lo,hi,symbol in ranges:
                if lo<=ip<hi:targeted.append((i,symbol,hex(ip)))
        for _,name,_ in set(targeted):counts[name]+=1
        if targeted:samples.append(dict(thread=lines[0].strip(),leaf=lines[1].strip(),path=targeted))
    return dict(mapping=mapping,counts=counts,samples=samples)


if __name__=="__main__":
    study=Path(sys.argv[1])
    Path(sys.argv[2]).write_text(json.dumps({leg:run(study,leg) for leg in ("01-A","02-B")},indent=2)+"\n")
