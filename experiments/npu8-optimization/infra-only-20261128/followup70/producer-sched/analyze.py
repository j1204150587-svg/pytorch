"""Intersect one Level1 trace's linked pre-ATen gap prefixes with that worker's perf clock."""
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path
import re
from statistics import median
import subprocess
import sys

HEADER=re.compile(r"^.*?\s(\d+)/(\d+)\s+(\d+\.\d+):\s+(.*)$")
FRAME=re.compile(r"^\s*[0-9a-f]+\s+(.+)\s+\((.*)\)\s*$")
def wall(x):return int(Decimal(str(x))*1000)
def read(p):return json.loads(p.read_bytes())
def dump(p,v):p.write_text(json.dumps(v,indent=2)+"\n")
def rows(p):
    with gzip.open(p,"rt",newline="") as f:return list(csv.DictReader(f))
def save_rows(p,values,fields):
    with gzip.open(p,"wt",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(values)
def event_list(p):
    events=[]
    with p.open(errors="replace") as f:
        for line in f:
            match=HEADER.match(line)
            if match:
                _,tid,seconds,label=match.groups()
                events.append((int(tid),int(Decimal(seconds)*1000000000),label.strip(),[]))
            elif events and (match:=FRAME.match(line)):
                events[-1][-1].append(match.groups())
    return events
def point_class(frames):
    if not frames:return "unknown"
    sym,dso=frames[0]
    library=dso.rsplit("/",1)[-1]
    if "profiler" in sym.lower() or "kineto" in sym.lower():return "profiler"
    if any(s in library for s in ("torch_npu","libacl","libascend","libopapi","libruntime","libplatform","libge_")):return "NPU substrate"
    if "libtorch_cpu" in library or "libc10.so" in library:return "PyTorch CPU"
    if "libtorch_python" in library:return "PyTorch Python binding"
    if library.startswith(("libpython","python3")):return "CPython interpreter"
    if library.startswith(("libc.","libstdc++","libgcc","ld-linux","libpthread","libm.")):return "libc/allocator/runtime"
    return "unknown"
def parse_anchor(events,thread,convert):
    pending={};pairs=defaultdict(list)
    for tid,t,tag,_ in events:
        if tid not in thread:continue
        if ":enter:" in tag:pending[tid]=convert(t)
        elif ":leave:" in tag and tid in pending:pairs[thread[tid]].append((pending.pop(tid),convert(t)))
    return pairs
def bounds(roots,probes):
    matches=[]
    for tid,spans in roots.items():
        pairs=probes[tid]
        for start,end in spans:
            # Match only a unique call with nearby endpoints on both clocks.
            neighbors=[(a,b) for a,b in pairs if max(abs(a-start),abs(b-end))<200000]
            if len(neighbors)!=1:continue
            a,b=neighbors[0]
            if sum(max(abs(c-a),abs(d-b))<200000 for c,d in spans)!=1:continue
            matches.append(dict(tid=tid,trace=[start,end],perf=[a,b],lower_ns=end-b,upper_ns=start-a))
    if not matches:return None,None,matches
    return max(r["lower_ns"] for r in matches),min(r["upper_ns"] for r in matches),matches
def classify_lanes(events,thread,convert,lost):
    switches=defaultdict(list);points=defaultdict(list)
    for tid,t,tag,frames in events:
        if tid not in thread:continue
        n=thread[tid];t=convert(t)
        if tag.startswith("PERF_RECORD_SWITCH "):
            switches[n].append((t,tag.split()[1],tag.endswith("preempt")))
        elif tag=="cpu-clock:u:":points[n].append((t,point_class(frames),frames))
    lanes={}
    for n,seq in switches.items():
        spans=[]
        for (left,s,preempt),(right,e,_) in zip(seq,seq[1:]):
            if s==e or lost:state="unknown"
            elif s=="IN":state="running"
            elif preempt:state="off_preempt"
            else:state="off_other"
            if right>left:spans.append((left,right,state))
        lanes[n]=(spans,[b for a,b,s in spans])
    return lanes,points
def overlap(a,b,ref,ends):
    totals=Counter();i=bisect_right(ends,a)
    while i<len(ref) and ref[i][0]<b:
        l,r,state=ref[i];totals[state]+=min(b,r)-max(a,l);i+=1
    totals["unknown"]+=b-a-sum(totals.values())
    return totals
def main():
    trace,where,cap=map(Path,sys.argv[1:])
    rec=read(cap/"receipt.json")
    thread={v["host"]:v["ns"] for v in rec["threads"].values()}
    clocks=[json.loads(l) for l in (cap/"clock.jsonl").read_text().splitlines()]
    offs=[(a+b)//2-m for a,m,r,b in clocks]
    center=sorted(offs)[len(offs)//2]
    clock_lo=min(a-m-center for a,m,r,b in clocks)
    clock_hi=max(b-m-center for a,m,r,b in clocks)
    # --clockid mono applies to switch records, timer samples and sync entry/return probes.
    report={"threads":rec["threads"],"clock":{"rtc_minus_mono_ns":center,"clock_lo_delta_ns":clock_lo,
            "clock_hi_delta_ns":clock_hi,"samples":len(clocks),"max_bracket_ns":max(b-a for a,m,r,b in clocks),
            "max_sample_separation_ns":max(clocks[i+1][1]-row[1] for i,row in enumerate(clocks[:-1]))}}
    anchor=event_list(cap/"anchor.txt");samples=event_list(cap/("sched-symbols.txt" if (cap/"sched-symbols.txt").exists() else "sched.txt"))
    with gzip.open(trace,"rt",encoding="utf-8-sig") as f:raw=json.load(f)
    ev=raw if isinstance(raw,list) else raw["traceEvents"];syncs=defaultdict(list);steps={}
    for e in ev:
        if e.get("ph")!="X":continue
        name=e["name"]
        if name=="AscendCL@aclrtSynchronizeDeviceWithTimeout" or name.startswith("ProfilerStep#"):
            a=wall(e["ts"]);b=a+wall(e["dur"])
            if name.startswith("ProfilerStep#"):steps[int(name.split("#")[-1])]=(a,b)
            else:syncs[int(e["tid"])].append((a,b))
    del raw,ev
    mono_wall=lambda t:t+center
    constant_low,constant_high,first=bounds(syncs,parse_anchor(anchor,thread,mono_wall))
    mon=[x[1] for x in clocks];delta=[x[2]-x[1] for x in clocks]
    def raw_delta(t):
        i=max(0,min(len(mon)-2,bisect_right(mon,t)-1))
        return delta[i]+(delta[i+1]-delta[i])*(t-mon[i])//(mon[i+1]-mon[i])
    ref=(first[0]["perf"][0]-center) if first else mon[0]
    baseline_delta=raw_delta(ref)
    aligned=lambda t:t+center+raw_delta(t)-baseline_delta
    low,high,matches=bounds(syncs,parse_anchor(anchor,thread,aligned))
    verified=low is not None and low<=high and len(matches)>=2
    bias=(low+high)//2 if verified else 0
    convert=lambda t:aligned(t)+bias
    neighbor=max(abs(b-a) for a,b in zip(delta,delta[1:]))
    correction_uncertainty=neighbor+max(b-a for a,m,r,b in clocks)+(clock_hi-clock_lo)
    assert min(t for a,b in steps.values() for t in (a,b))>=convert(mon[0]) and max(b for a,b in steps.values())<convert(mon[-1])
    report["clock"].update(constant_wall_lower=constant_low,constant_wall_upper=constant_high,raw_minus_mono_change_ns=delta[-1]-delta[0],
                           conversion_reference_mono=ref,raw_minus_mono_reference=baseline_delta,raw_neighbor_max_ns=neighbor,
                           raw_minus_mono_nonincreasing=all(a>=b for a,b in zip(delta,delta[1:])),fitted_midpoint_ns=bias)
    report["clock"].update(anchor_pairs=matches,anchor_verified=verified,perf_to_trace_lower=low,perf_to_trace_upper=high)
    # Width of the observed RTC/mono offset series also accounts for drift inside the recorded window.
    perf_error=max(abs(low-bias),abs(high-bias))+correction_uncertainty if verified else None
    report["clock"]["perf_trace_error_ns"]=perf_error
    device=read(where/"device-clock.json")
    lost=False
    for basename in ("sched","anchor"):
        matches_lost=[]
        cmd=["perf","script","-D","-i",str(cap/(basename+".data"))]
        with subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,errors="replace") as p:
            for line in p.stdout:
                if "PERF_RECORD_LOST" in line:matches_lost.append(line.strip())
        report[basename+"_lost_records"]=matches_lost
        lost |= bool(matches_lost)
    lanes,points=classify_lanes(samples,thread,convert,lost)
    point_times={n:[v[0] for v in data] for n,data in points.items()}
    report["switch_events"]={n:len(lane[0])+1 for n,lane in lanes.items()}
    report["sample_points"]={n:len(p) for n,p in points.items()}
    calls={r["queue_id"]:r for r in rows(where/"queue_calls.csv.gz")}
    obs=defaultdict(lambda:dict(sum=Counter(),samples=Counter(),leaf=Counter(),exposures=0))
    detailed=[]; hits=[]
    for g in rows(where/"device_gaps.csv.gz"):
        if g["state"]!="linked" or not int(g.get("before_aten_ns") or 0):continue
        step=int(g["step"]);left=steps[step][0]+int(g["step_offset_ns"])
        c=calls[g["queue_id"]];tid=int(c["producer_tid"])
        at=steps[step][0]+int(c["step_offset_ns"])-int(c["aten_to_enqueue_ns"])
        right=left+int(g["gap_ns"])
        assert min(right,at)-left==int(g["before_aten_ns"])
        clk=device["ProfilerStep#"+str(step)]
        row={k:g[k] for k in ("step","batch","seq_len","stream","queue_id","task","op","role","gap_ns")}
        row.update(tid=tid,aten=c["aten"],left=left,right=right,aten_start=at)
        for mode in ("nominal","robust"):
            if mode=="nominal":a,b=left,min(right,at)
            elif not verified or not clk["consistent"] or clk["clock_upper_ns"] is None or lost:a=b=left
            else:
                a=left+clk["clock_upper_ns"]+perf_error
                b=max(a,min(right+clk["clock_lower_ns"],at)-perf_error)
            state=overlap(a,b,*lanes.get(tid,([],[])))
            r=obs[step,c["role"],mode];r["sum"].update(state);r["exposures"]+=int(a<b)
            row[mode+"_ns"]=b-a
            for name in ("running","off_preempt","off_other","unknown"):
                row[mode+"_"+name]=state[name]
            times=point_times.get(tid,[])
            for i in range(bisect_left(times,a),bisect_left(times,b)):
                t,cls,frames=points[tid][i];r["samples"][cls]+=1
                if frames:
                    symbol,library=frames[0];r["leaf"][library.rsplit("/",1)[-1]+"!"+symbol]+=1
                hits.append(dict(step=step,mode=mode,role=c["role"],queue_id=g["queue_id"],t=t,
                                 offset_from_gap_ns=t-left,cls=cls,stack=json.dumps(frames,ensure_ascii=False)))
        detailed.append(row)
    save_rows(where/"prefixes.csv.gz",detailed,list(detailed[0]))
    save_rows(where/"samples-in-prefix.csv.gz",hits,["step","mode","role","queue_id","t","offset_from_gap_ns","cls","stack"])
    for r in obs.values():r["leaf"]=dict(r["leaf"].most_common(16))
    report["steps"]=dict((f"{s}-{role}-{mode}",r) for (s,role,mode),r in sorted(obs.items()))
    report["largest"]=sorted(detailed,key=lambda x:-x["robust_ns"])[:12]
    for step,(a,b) in steps.items():
        report.setdefault("step_wall",{})[step]={"a":a,"b":b,"duration_ns":b-a}
    dump(where/"sched-analysis.json",report)

if __name__=="__main__":main()

