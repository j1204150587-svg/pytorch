"""Count device-context calls in baseline CPU-clock stacks from the owning process maps."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("perfdir",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    result={}
    for leg in ("01-A","02-B"):
        w=json.loads((args.perfdir/"pair"/leg/"worker.json").read_bytes())
        root=Path("/proc")/str(w["init_pid"])/"root"
        data=args.perfdir/(leg+".data")
        maps=subprocess.check_output(["perf","script","-i",data,"--show-mmap-events","--show-task-events"],
                                     text=True,stderr=subprocess.DEVNULL)
        bases={}
        for m in re.finditer(r"PERF_RECORD_MMAP2 [^:]+: \[(0x[\da-f]+)\((0x[\da-f]+)\) @ ([\da-fx]+) [^\]]+\]: r-xp (.+)",maps):
            name=m[4].strip()
            if name.endswith(("/libc.so.6","/libtorch_npu.so")):
                bases[name]=(int(m[1],16),int(m[2],16),int(m[3],0))
        assert len(bases)==2
        raw=subprocess.check_output(["perf","script","-i",data,"--symfs",str(root),"-F","comm,pid,tid,ip,sym,dso"],
                                    text=True,stderr=subprocess.DEVNULL)
        frames=[]
        requests={k:set() for k in bases}
        sample=[]
        for line in raw.splitlines():
            if not line.strip(): continue
            if not line.startswith("\t"):
                if sample: frames.append(sample)
                sample=[]
                continue
            m=re.match(r"\s*([0-9a-f]+) .*? \((.*)\)\s*$",line)
            pair=None
            if m and m[2] in bases:
                name=m[2]
                start,length,offset=bases[name]
                addr=int(m[1],16)
                if start<=addr<start+length:
                    addr=addr-start+offset
                    pair=name,addr
                    requests[name].add(addr)
            sample.append(pair)
        if sample: frames.append(sample)
        lookup={}
        for name,items in requests.items():
            items=sorted(items)
            syms=subprocess.check_output(["addr2line","-C","-f","-e",root/name.lstrip("/")],
                                         input="".join(f"0x{x:x}\n" for x in items),text=True).splitlines()[::2]
            assert len(syms)==len(items)
            lookup.update({(name,addr):sym for addr,sym in zip(items,syms)})
        counts=Counter()
        leaves=Counter()
        callers=Counter()
        for stack in frames:
            names=[lookup.get(frame,"") for frame in stack]
            counts["all"]+=1
            idx=next((i for i,s in enumerate(names) if "c10_npu::isDeviceCtxActive(" in s),None)
            if idx is None:continue
            counts["ctx_in_stack"]+=1
            counts["maybe_set_device_ancestor"]+=any("c10_npu::MaybeSetDevice(" in s for s in names[idx+1:])
            counts["mutex_leaf_and_ctx"]+="pthread_mutex_lock" in names[0]
            counts["check_compatible_child"]+=any("CheckCompatibleImpl(" in s for s in names[:idx])
            leaves[names[0] or "unresolved_or_other_dso"]+=1
            callers[names[idx+1] if idx+1<len(names) else "unknown"]+=1
        result[leg]=dict(counts=counts,leaves=leaves,direct_callers=callers)
    args.out.write_text(json.dumps(result,indent=2)+"\n")

if __name__=="__main__": main()
