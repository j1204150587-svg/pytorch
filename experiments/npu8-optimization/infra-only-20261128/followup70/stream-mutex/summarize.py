"""Summarize the hottest active stream without merging address spaces across ranks."""
from collections import Counter
import csv
import json
from pathlib import Path
import statistics


here=Path(__file__).resolve().parent
report={}
for leg,data in json.loads((here/"analysis.json").read_bytes()).items():
    streams=data["streams"]
    addr=max(streams,key=lambda a:sum(streams[a].get("s",{}).get("incoming_words",{}).values()))
    stream=streams[addr]
    hot={}
    for kind in ("s","r"):
        d=stream[kind]
        hot[kind]=dict(threads=d["threads"],incoming_words=d["incoming_words"],outgoing_words=d["outgoing_words"],
                       cpu_top=Counter(d["lock_cpus"]).most_common(3),futex=d["futex"],
                       measured_acquisition_ns=d["acquisition_ns"],measured_held_ns=d["held_ns"])
    arm_counts=Counter()
    for pkt in data["spe"]["hits"]:
        if addr not in pkt["object_offsets"]: continue
        offset=pkt["object_offsets"][addr]
        if offset in {40,48,52,56,88,96,100,104}:
            arm_counts[f"{pkt['comm']}:{pkt['operation']}:+{offset}"]+=1
    cnt=Counter()
    for streams_in_rank in data["glibc_entries"]["by_object"].values():
        for k,values in streams_in_rank.items():
            if k=="s":
                for key,value in values.items():
                    cnt[key.split(":word=")[1].split(":")[0]+":"+key.split(":")[2]]+=value
    report[leg]=dict(obj=addr,obj_mod128=int(addr,16)%128,private_cache_shared=stream["same_64B_line_hot_fields"],
        llc_line_shared=stream["same_128B_line_hot_fields"],total_wrapper_events=data["counts"],observed_streams=len(streams),
        process_futex_enters=data["futex_enters"],hottest=hot,glibc_stream_words_all=cnt,
        glibc_calls=data["glibc_entries"]["calls"],glibc_hottest=data["glibc_entries"]["by_object"].get(addr),
        spe_user_ldst=data["spe"]["totals"],packets_near_observed_streams=data["spe"]["matching_packet_count"],
        spe_hot_lock_fields=arm_counts)
with (here/"placement-cpu.csv").open() as f: rows=list(csv.DictReader(f))
report["cpu_forced_handoff"]={}
for cpu in ("188","266"):
    groups={}
    for layout in ("96","64"):
        group=[r for r in rows if r["submit_cpu"]==cpu and r["base_mod128"]==layout]
        groups[layout]={k:statistics.median(float(r[k]) for r in group)
                        for k in ["first_p50_ns","first_p95_ns","batch_p50_ns","batch_p95_ns"]}
    report["cpu_forced_handoff"][cpu]=groups
(here/"measurement-summary.json").write_text(json.dumps(report,indent=2)+"\n")
