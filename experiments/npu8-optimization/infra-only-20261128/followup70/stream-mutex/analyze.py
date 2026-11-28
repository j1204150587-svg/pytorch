"""Reconstruct per-stream lock ownership, futex observations and same-line ARM SPE accesses."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
OFF = {"s": 0x28, "r": 0x58}


def hist(values):
    if not values:
        return {"n": 0}
    values = sorted(values)
    return dict(n=len(values), **{name: values[int(q * (len(values)-1))]
                for q,name in [(0.5,"p50"),(0.95,"p95"),(0.99,"p99"),(1,"max")]})


def parse_locks(path, worker):
    group = "hstu_npu8_stream_20261128"
    pattern = re.compile(r"^\s*(.+?)\s+(\d+)/(\d+)\s+\[(\d+)\]\s+(\d+)\.(\d+):\s+(\S+):\s+(.*)$")
    extract_field = re.compile(r"(addr|word|caller|rc)=((?:0x)?[\da-f]+)")
    counts = Counter()
    locks = defaultdict(lambda: dict(threads=Counter(), lock_cpus=Counter(), incoming_words=Counter(), outgoing_words=Counter(),
                                    acquisition_ns=[], held_ns=[], callers=Counter(), futex=Counter()))
    held = {}
    begin = {}
    futex = []
    seen_futex = {}
    tids = set()
    for original in path.read_text().splitlines():
        match = pattern.match(original)
        if not match:
            if original.strip(): counts["unparsed_lines"] += 1
            continue
        comm,pid,tid,cpu,secs,ns,kind,body = match.groups()
        if int(pid) != worker["pid"]:
            counts["other_tgid"] += 1
            continue
        tids.add(tid)
        ts = int(secs)*10**9 + int(ns.ljust(9,"0"))
        if kind.startswith("syscalls:"):
            if kind.endswith("sys_enter_futex"):
                data = dict((k,int(n,16)) for k,n in re.findall(r"(uaddr|op|val): 0x([\da-f]+)",body))
                if "uaddr" not in data:
                    # perf versions printing syscall fields as named function arguments.
                    data = dict((k,int(n,16)) for k,n in re.findall(r"\b(uaddr|op|val):\s*([\da-f]+)",body))
                if "uaddr" in data and "op" in data:
                    seen_futex[tid] = (ts,data["uaddr"],data["op"] & 0x7f,comm)
                    futex.append((data["uaddr"], data["op"] & 0x7f, "enter", comm, 0))
                else: counts["unparsed_enter_futex"] += 1
            elif tid in seen_futex:
                entry,addr,op,fc = seen_futex.pop(tid)
                futex.append((addr,op,"return",fc,ts-entry))
            continue
        if not kind.startswith(group+":"):
            counts["other_event"] += 1
            continue
        event = kind.split(":")[-1]
        cls, action = event.split("_")
        values = dict((k,int(v,0)) for k,v in extract_field.findall(body))
        counts[event] += 1
        if action == "begin":
            addr = values["addr"]
            rec = locks[(cls,addr)]
            rec["threads"][f"{tid}:{comm}:lock"] += 1
            rec["lock_cpus"][f"{tid}:{cpu}"] += 1
            rec["incoming_words"][str(values["word"])] += 1
            begin[tid,cls] = (ts,addr)
        elif action == "got":
            if (tid,cls) in begin:
                started,addr = begin.pop((tid,cls))
                if values["rc"] == 0:
                    locks[(cls,addr)]["acquisition_ns"].append(ts-started)
                    held[tid,cls,addr] = ts
        else:
            addr = values["addr"]
            rec = locks[(cls,addr)]
            rec["threads"][f"{tid}:{comm}:unlock"] += 1
            rec["outgoing_words"][str(values["word"])] += 1
            rec["callers"][hex(values["caller"])] += 1
            if (tid,cls,addr) in held:
                rec["held_ns"].append(ts-held.pop((tid,cls,addr)))
    assert counts["s_begin"] > 200 and counts["s_got"] > 200 and counts["s_end"] > 200, counts
    assert counts["unparsed_enter_futex"] == 0, counts
    alladdrs = {addr: (cls,addr) for cls,addr in locks}
    for addr,op,action,comm,duration in futex:
        if addr in alladdrs:
            rec = locks[alladdrs[addr]]
            rec["futex"][f"{comm}:{op}:{action}"] += 1
    streams = {}
    lines = defaultdict(set)
    for (cls, addr), info in locks.items():
        obj = addr - OFF[cls]
        # glibc normal mutex: __lock at +0, __owner at +8 and __nusers at +12.
        a,b = obj+OFF["s"],obj+OFF["r"]
        for x in range((a//128)*2,((b+12)//128+1)*2):
            lines[x].add(obj)
        rec = streams.setdefault(hex(obj), dict(obj_mod64=obj%64,stream_word=hex(a),recycle_word=hex(b),
                         same_64B_line_lock_words=a//64==b//64,
                         same_128B_line_lock_words=a//128==b//128,
                         same_128B_line_hot_fields=bool({(a+x)//128 for x in (0,8,12)}
                                                        & {(b+x)//128 for x in (0,8,12)}),
                         same_64B_line_hot_fields=bool({(a+x)//64 for x in (0,8,12)}
                                                       & {(b+x)//64 for x in (0,8,12)})))
        info["acquisition_ns"] = hist(info["acquisition_ns"])
        info["held_ns"] = hist(info["held_ns"])
        rec[cls] = info
    return dict(counts=counts, recorded_threads=len(tids), futex_enters=sum(row[2]=="enter" for row in futex),
                streams=streams), lines


def parse_spe(path, worker, lines):
    """Read the PMU packet stream to retain operation type and data VA (one row per completed packet)."""
    ranges = []
    for row in worker["maps"]:
        region,perms,off,_,_,dso = row.split(maxsplit=5)
        if "x" in perms:
            lo,hi = [int(x,16) for x in region.split("-")]
            ranges.append((lo,hi,int(off,16),Path(dso).name))
    def pc(fileaddr):
        return next((name+"+"+hex(fileaddr-lo+off) for lo,hi,off,name in ranges if lo<=fileaddr<hi),hex(fileaddr))
    proc = subprocess.Popen(["perf","report","-D","-i",str(path)], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL)
    record = {}
    total = Counter()
    hits = []
    packet = re.compile(r"^\.\s+[\da-f]+:\s+(?:[\da-f]{2} )+\s+([A-Z]+)\s*(.*)")
    for row in proc.stdout:
        m = packet.match(row)
        if not m: continue
        key,value=m.groups()
        if key=="PC" and re.match(r"^0x[\da-f]+ el[0-3] ",value):
            record = dict(pc=int(value.split()[0],16),el=value.split()[1])
        elif key=="VA" and record and re.fullmatch("0x[0-9a-f]+",value):
            record["va"]=int(value.split()[0],16)
        elif key in ("LD","ST") and record:
            record["op"]=key
        elif key=="EV" and record:
            record["events"]=value
        elif key=="CONTEXT" and record and re.match("^0x[0-9a-f]+ el",value):
            record["tid"]=int(value.split()[0],16)
        elif key=="LAT" and record and value.endswith(" TOT"):
            record["total_latency_cycles"]=int(value.split()[0])
        elif key in ("TS","END") and record:
            if all(k in record for k in ("va","tid","op")) and record["el"] == "el0":
                total["user_ldst"] += 1
                known=str(record["tid"]) in worker["tids"]
                if not known: total["unknown_thread_at_snapshot"] += 1
                objs=lines.get(record["va"]//64) if known else None
                if objs:
                    site=pc(record["pc"])
                    hits.append(dict(tid=record["tid"],comm=worker["tids"].get(str(record["tid"]),"unknown"),
                                    operation=record["op"],site=site,va=hex(record["va"]),
                                    object_offsets={hex(obj):record["va"]-obj for obj in objs},
                                    events=record.get("events"),latency_cycles=record.get("total_latency_cycles")))
            record={}
    assert proc.wait()==0 and total["user_ldst"]>100, total
    return dict(totals=total,matching_packet_count=len(hits),hits=hits)


def all_glibc_callers(path, streams):
    pattern=re.compile(r"^\s*(.+?)\s+(\d+)/(\d+)\s+\[(\d+)\]\s+\S+:\s+"
                       r"hstu_npu8_stream_all_20261128:(lock|unlock):.*?addr=(0x[\da-f]+)\s+"
                       r"word=(\d+)\s+caller=(0x[\da-f]+)")
    results={}
    for obj,stream in streams.items():
        results[stream["stream_word"]]=(obj,"s")
        results[stream["recycle_word"]]=(obj,"r")
    events=defaultdict(lambda:defaultdict(Counter))
    n=0
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        m=pattern.search(line)
        assert m,line
        comm,pid,tid,cpu,which,addr,word,caller=m.groups()
        obj,cls=results[hex(int(addr,16))]
        k=events[obj][cls]
        k[f"{comm}:{tid}:{which}:word={word}:caller={caller}"]+=1
        n+=1
    assert n>50
    return dict(calls=n,by_object=events)


def main():
    result={}
    for leg in ("01-A","02-B"):
        worker=json.loads((HERE/(leg+".worker.json")).read_bytes())
        data,lines=parse_locks(HERE/(leg+".locks.txt"),worker)
        data["spe"]=parse_spe(HERE/(leg+".spe.data"),worker,lines)
        data["glibc_entries"]=all_glibc_callers(HERE/(leg+".all.txt"),data["streams"])
        result[leg]=data
    (HERE/"analysis.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__": main()
