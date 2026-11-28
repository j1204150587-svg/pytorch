"""Count observed cross-thread/cross-CPU lock-end interleavings on each hot instance."""
from collections import Counter
import json
from pathlib import Path
import re
import sys
import tarfile


here=Path(__file__).resolve().parent
summary=json.loads((here/"measurement-summary.json").read_bytes())
pattern=re.compile(r"^\s*(.+?)\s+\d+/(\d+)\s+\[(\d+)\]\s+(\d+)\.(\d+):\s+"
                   r"hstu_npu8_stream_20261128:([sr])_(begin|end):.*?addr=(0x[\da-f]+)")
result={}
with tarfile.open(here/"evidence.tar.gz") as arc:
    for leg in ("01-A","02-B"):
        obj=int(summary[leg]["obj"],16)
        last=None
        latest_submit=None
        ended_recycles=[]
        classified=Counter()
        sequences=Counter()
        for raw in arc.extractfile(leg+".locks.txt"):
            m=pattern.search(raw.decode())
            if not m: continue
            comm,tid,cpu,secs,ns,cls,action,addr=m.groups()
            if int(addr,16) != obj+(0x28 if cls=="s" else 0x58): continue
            ts=int(secs)*10**9+int(ns.ljust(9,"0"))
            if action=="end":
                if last:
                    other,other_cpu=last
                    sequences[other+"→"+cls+(":same_cpu" if other_cpu==cpu else ":cross_cpu")]+=1
                last=cls,cpu
            if cls=="s" and action=="end":
                latest_submit=ts
                ended_recycles=[]
            if cls=="r" and action=="end" and latest_submit is not None:
                ended_recycles.append((ts,cpu))
            if cls=="s" and action=="begin" and latest_submit is not None:
                classified["submit_begins_after_observed_end"]+=1
                if ended_recycles:
                    classified["recycle_end_between_prior_submit_end_and_next_submit_begin"]+=1
                    classified["different_cpu_on_last_recycle_end"]+=ended_recycles[-1][1]!=cpu
                # Count one inter-submit interval once even if a later record was lost.
                latest_submit=None
                ended_recycles=[]
        result[leg]=dict(hottest_address=hex(obj),counts=classified,consecutive_observed_unlocks=sequences)
(here/"handoff.json").write_text(json.dumps(result,indent=2)+"\n")
