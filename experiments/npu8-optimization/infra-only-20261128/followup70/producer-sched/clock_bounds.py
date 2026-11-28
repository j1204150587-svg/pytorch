"""Bound a constant device-to-host clock correction from real submissions and global drains."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import sys

cli = argparse.ArgumentParser()
cli.add_argument("trace", type=Path)
cli.add_argument("analyzer", type=Path)
cli.add_argument("output", type=Path)
args = cli.parse_args()
sys.path.insert(0, str(args.analyzer.parent))
from analyze_dynamic_trace import Lanes, Span, is_work, ns

with gzip.open(args.trace,"rt",encoding="utf-8-sig") as f:
    raw=json.load(f)
events=raw if isinstance(raw,list) else raw["traceEvents"]
pids={e["args"]["name"]:e["pid"] for e in events if e["name"]=="process_name"}
launches,drains,tasks,steps=[],[],[],{}
starts,ends,anchors=defaultdict(list),defaultdict(list),defaultdict(list)
for e in events:
    ph,cat,pid,name=e["ph"],e.get("cat"),e["pid"],e["name"]
    if cat=="HostToDevice":
        if ph=="s" and pid==pids["CANN"]:
            starts[e["id"]].append((e["tid"],ns(e["ts"])))
        if ph=="f" and pid==pids["Ascend Hardware"]:
            ends[e["tid"],ns(e["ts"])].append(e["id"])
    if ph!="X": continue
    s=Span.read(e)
    if name.startswith("ProfilerStep#"):steps[name]=s
    if pid==pids["CANN"]:
        anchors[s.tid,s.start,s.args.get("connection_id")].append(s)
        if name.startswith(("AscendCL@aclrtLaunchKernel","AscendCL@aclrtMemcpy")):launches.append(s)
        if name=="AscendCL@aclrtSynchronizeDeviceWithTimeout":drains.append(s)
    if pid==pids["Ascend Hardware"] and is_work(s.args.get("Task Type","")):tasks.append(s)
index=Lanes(launches)
linked=[]
for t in tasks:
    ids=ends[t.tid,t.start]
    if len(ids)!=1 or len(starts[ids[0]])!=1:continue
    tid,start=starts[ids[0]][0]
    a=anchors[tid,start,t.args.get("connection_id")]
    if len(a)!=1:continue
    launched=index.children(a[0])
    if len(launched)==1:linked.append((launched[0],t))
by_start=defaultdict(list)
for t in tasks: by_start[t.start].append(t)
holes={}
frontier=None
for start, batch in sorted(by_start.items()):
    if frontier is not None and frontier<start and len(batch)==1:
        holes[id(batch[0])]=(frontier,start)
    frontier=max(frontier or 0, max(t.end for t in batch))
out={}
for name,step in sorted(steps.items()):
    group=[(a,t) for a,t in linked if step.start<=t.start<t.end<=step.end and step.start<=a.start<step.end]
    lower=max(a.start-t.start for a,t in group)
    constraints=[]
    for s in drains:
        earlier=[t.end for a,t in group if a.end<s.start]
        if earlier: constraints.append({"drain_end_ns":s.end,"upper_ns":s.end-max(earlier)})
    upper=min((r["upper_ns"] for r in constraints),default=None)
    # Shift the two ends of the same all-stream idle interval together.
    # Only linked physical successors with a gap fully inside this step contribute.
    selected=[(a,holes[id(t)]) for a,t in group if id(t) in holes and holes[id(t)][0]>=step.start]
    def already_returned(offset):
        return round(sum(max(0,right+offset-max(left+offset,a.end)) for a,(left,right) in selected)/1e6,6)
    out[name]={"tasks_with_unique_acl":len(group),"clock_lower_ns":lower,
               "clock_upper_ns":upper,"global_drains":constraints,
               "linked_gaps":len(selected),
               "linked_gap_ms":round(sum(r-l for _,(l,r) in selected)/1e6,6),
               "after_return_ms":{"earliest":already_returned(lower),"nominal":already_returned(0),
                                  "latest":already_returned(upper) if upper is not None else None},
               "consistent":upper is None or lower<=upper}
args.output.write_text(json.dumps(out,indent=2))
