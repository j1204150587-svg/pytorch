"""Resolve ASLR-normalized user IPs against the recorded container's ELF ranges."""
from collections import defaultdict
from pathlib import Path
import re
import subprocess
import sys

MM = re.compile(r"PERF_RECORD_MMAP2 [^:]+: \[(0x[\da-f]+)\((0x[\da-f]+)\) @ ([\da-fx]+) [^\]]+\]: r-xp (.+)")
FRAME = re.compile(r"^(\s*)([0-9a-f]+) (.*?) \((.*)\)$")
def main():
    cap,root=map(Path,sys.argv[1:])
    mappings=defaultdict(list)
    with (cap/"mappings.txt").open("w") as out:
        cmd=["perf","script","--no-inline","--show-mmap-events","--show-task-events","-i",str(cap/"sched.data")]
        with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True) as p:
            for line in p.stdout:
                m=MM.search(line)
                if not m:continue
                out.write(line)
                l,n,f=map(lambda x:int(x,0),m.group(1,2,3));mappings[m[4].strip()].append((l,l+n,f))
    lines=(cap/"sched.txt").read_text().splitlines()
    addresses=defaultdict(set);frames={}
    for i,line in enumerate(lines):
        m=FRAME.match(line)
        if not m or m[3]!="[unknown]":continue
        ip=int(m[2],16);lib=m[4]
        for lo,hi,off in mappings[lib]:
            if lo<=ip<hi:
                fo=ip-lo+off
                addresses[lib].add(fo);frames[i]=(m[1],m[2],lib,fo);break
    names={}
    for lib,offsets in addresses.items():
        elf=root/lib.lstrip("/")
        if not elf.is_file():continue
        segs=[]
        program=subprocess.run(["readelf","-Wl",elf],capture_output=True,text=True)
        for line in program.stdout.splitlines():
            cols=line.split()
            if cols and cols[0]=="LOAD" and "E" in cols[6:-1]:
                off,virt,sz=int(cols[1],16),int(cols[2],16),int(cols[4],16)
                segs.append((off,off+sz,virt))
        work=[]
        for f in sorted(offsets):
            for l,r,vma in segs:
                if l<=f<r:work.append((f,f-l+vma));break
        if not work:continue
        addr=subprocess.run(["addr2line","-C","-f","-e",elf],input="".join(hex(a)+"\n" for f,a in work),
                             capture_output=True,text=True)
        results=addr.stdout.splitlines()
        if len(results)!=2*len(work):continue
        for (f,vma),sym in zip(work,results[::2]):
            names[lib,f]=sym if sym!="??" else f"[ELF+{vma:#x}]"
    for i,(indent,ip,lib,f) in frames.items():lines[i]=indent+ip+" "+names.get((lib,f),f"[file+{f:#x}]")+" ("+lib+")"
    (cap/"sched-symbols.txt").write_text("\n".join(lines)+"\n")
if __name__=="__main__":main()

