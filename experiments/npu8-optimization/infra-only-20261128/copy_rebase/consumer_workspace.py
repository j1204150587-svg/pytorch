"""Pair Level1 single-SDMA calls for which B moved GetWorkspaceSize into its own dequeue interval."""
import argparse
from bisect import bisect_left
from collections import defaultdict
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path


def ns(value):
    return int(Decimal(str(value))*1000)


def csv_rows(folder,name):
    with gzip.open(folder/(name+".csv.gz"),"rt",newline="") as stream:return list(csv.DictReader(stream))


def collect(raw,folder):
    with gzip.open(raw/"trace_view.json.gz","rt",encoding="utf-8-sig") as src: trace=json.load(src)
    events=trace if isinstance(trace,list) else trace["traceEvents"]
    starts,workspaces,dequeues={},defaultdict(list),{}
    for e in events:
        if e.get("ph")!="X":continue
        name=e.get("name","")
        if name.startswith("ProfilerStep#"):
            starts[name.split("#")[-1]]=ns(e["ts"])
        elif name=="AscendCL@aclnnInplaceCopyGetWorkspaceSize":
            t=ns(e["ts"])
            workspaces[e["tid"]].append((t,t+ns(e["dur"])))
        elif name=="Dequeue@aclnnInplaceCopy":
            t=ns(e["ts"])
            dequeues[str(e["args"]["correlation_id"])]=[e["tid"],t,t+ns(e["dur"])]
    del events,trace
    for spans in workspaces.values():spans.sort()
    times={tid:[s[0] for s in spans] for tid,spans in workspaces.items()}
    tasks=defaultdict(list)
    for row in csv_rows(folder,"work_tasks"):tasks[row["step"],row["queue_id"]].append(row)
    gaps={r["queue_id"]:r for r in csv_rows(folder,"device_gaps") if r["state"]=="linked" and r["op"]=="aclnnInplaceCopy"}
    out={}
    for r in csv_rows(folder,"queue_calls"):
        if r["op"]!="aclnnInplaceCopy" or r["aten"]!="aten::copy_":continue
        step,cid=r["step"],r["queue_id"]
        owned=tasks[step,cid]
        if len(owned)!=1 or owned[0]["task"]!="MEMCPY_ASYNC" or not owned[0]["task_type"].startswith("SDMA"):continue
        assert r["acl_api_sequence"]=="AscendCL@aclrtMemcpyAsync"
        tid,lo,hi=dequeues[cid]
        start=bisect_left(times.get(tid,[]),lo)
        spans=[]
        for a,b in workspaces[tid][start:]:
            if a>hi:break
            if b<=hi:spans.append((a,b))
        assert len(spans)<=1,(step,cid,spans)
        consumer_ns=sum(b-a for a,b in spans)
        assert not (consumer_ns and r["workspace_ns"])
        g=gaps.get(cid)
        ws_gap=0
        if g and spans:
            left=starts[step]+int(g["step_offset_ns"])
            right=left+int(g["gap_ns"])
            ws_gap=sum(max(0,min(right,b)-max(left,a)) for a,b in spans)
        key=(step,owned[0]["stream"],r["role"],r["aten_ordinal_in_thread"])
        assert key not in out
        out[key]=dict(cid=cid,slot=owned[0]["stream_work_slot"],producer_ws=int(r["workspace_ns"] or 0),
            consumer_ws=consumer_ns,consumer_ws_in_gap=ws_gap,producer_ws_in_gap=int(g["workspace_ns"] or 0) if g else 0,
            gap=int(g["gap_ns"]) if g else 0,aten_to_enqueue=int(r["aten_to_enqueue_ns"]),
            enqueue=int(r["enqueue_ns"]),queue_wait=int(r["signed_queue_wait_ns"]),dequeue=int(r["dequeue_ns"]),
            acl_launch=int(r["acl_api_sum_ns"]))
    return out


def distribution(items):
    vals=sorted(items)
    return dict(n=len(vals),sum_ms=sum(vals)/1e6,**{name:vals[int((len(vals)-1)*frac)]/1000
        for name,frac in (("p50_us",.5),("p95_us",.95),("p99_us",.99),("max_us",1.0))})


def main():
    p=argparse.ArgumentParser()
    p.add_argument("raw",type=Path)
    p.add_argument("analysis",type=Path)
    args=p.parse_args()
    all_rows={k:collect(args.raw/s,args.analysis/k) for k,s in (("A","01-A"),("B","02-B"))}
    moved=[key for key,r in all_rows["B"].items() if r["consumer_ws"]]
    assert moved and all(key in all_rows["A"] and all_rows["A"][key]["slot"]==all_rows["B"][key]["slot"] for key in moved)
    result={}
    for step,stream in sorted({k[:2] for k in moved}):
        keys=[k for k in moved if k[:2]==(step,stream)]
        result[f"{step}/stream{stream}"]={}
        for v in ("A","B"):
            rows=[all_rows[v][key] for key in keys]
            result[f"{step}/stream{stream}"][v]=dict(n=len(keys),metrics={f:distribution([r[f] for r in rows]) for f in rows[0] if f not in ("cid","slot")})
    (args.analysis/"moved_workspace.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":
    main()
