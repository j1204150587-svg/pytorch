"""Match IPG copy operations to their physical timeline and surrounding work."""
import argparse
import csv
from collections import Counter
from decimal import Decimal
import gzip
import json
from pathlib import Path


def ns(value):
    return int(Decimal(str(value))*1000)


def rows(path):
    with gzip.open(path, "rt", encoding="utf8") as stream:
        return list(csv.DictReader(stream))


def name(task):
    return (task["stream"], task["slot"], task["name"])


def cpu_overlap(events, scope, left, right):
    counts=Counter()
    intervals=[]
    for v in events:
        if (v.get("ph")!="X" or v["tid"]!=scope["tid"] or v.get("cat")!="cpu_op" or
            not v["name"].startswith("aten::")):
            continue
        a=ns(v["ts"])
        b=a+ns(v["dur"])
        if a<right and b>left:
            counts[v["name"]]+=1
            intervals.append((max(a,left),min(b,right)))
    cursor=left
    union=0
    for a,b in sorted(intervals):
        union+=max(0,b-max(a,cursor))
        cursor=max(b,cursor)
    return dict(main_thread_aten_counts=counts,aten_union_ms=union/1e6,outside_aten_ms=(right-left-union)/1e6)


def digest(leg, analysis):
    with gzip.open(leg / "trace_view.json.gz", "rt", encoding="utf-8-sig") as handle:
        loaded = json.load(handle)
    events = loaded if isinstance(loaded, list) else loaded["traceEvents"]
    hw = next(e["pid"] for e in events if e["name"]=="process_name" and e["args"]["name"]=="Ascend Hardware")
    physical = [e for e in events if e.get("ph")=="X" and e.get("pid")==hw and "Task Id" in e["args"]]
    markers = [e for e in events if e["name"]=="NPUIPGBucketRangeCopy" and e["ph"]=="X"]
    scopes = [e for e in events if e["name"].startswith("ProfilerStep#") and e["ph"]=="X"]
    deqs = {str(e["args"]["correlation_id"]):e for e in events if e.get("cat")=="enqueue" and e.get("ph")=="X"}
    queue = rows(analysis/"queue_calls.csv.gz")
    tasks = rows(analysis/"work_tasks.csv.gz")
    gaps = rows(analysis/"device_gaps.csv.gz")
    result=[]
    for scope in scopes:
        step=scope["name"].split("#")[1]
        start=ns(scope["ts"])
        end=start+ns(scope["dur"])
        calls = {r["queue_id"]:r for r in queue if r["step"]==step}
        times = {(str(e["args"]["Physic Stream Id"]),str(e["args"]["Task Id"]),e["name"]):
                 (ns(e["ts"]),ns(e["ts"])+ns(e["dur"])) for e in physical if start<=ns(e["ts"])<end}
        lane=[]
        for r in tasks:
            if r["step"]!=step:
                continue
            begin, finish=times[r["stream"],r["task_id"],r["task"]]
            lane.append(dict(start=begin,end=finish,name=r["task"],stream=r["stream"],slot=int(r["stream_work_slot"]),qid=r["queue_id"]))
        lane.sort(key=lambda t:(t["start"],t["name"]))
        idx=[i for i,t in enumerate(lane) if t["stream"]=="44" and calls.get(t["qid"],{}).get("aten")=="aten::copy_"
             and calls[t["qid"]]["role"]=="main"]
        assert idx==list(range(idx[0],idx[-1]+1)), "copy series overlaps other device submissions"
        prev, nxt = lane[idx[0]-1], lane[idx[-1]+1]
        window=lane[idx[0]:idx[-1]+1]
        assert all(t["start"]>=a["end"] for a,t in zip([prev]+window,window+[nxt]))
        cids={t["qid"] for t in window}
        parts=[g for g in gaps if g["step"]==step and g["queue_id"] in cids]
        durations=[t["end"]-t["start"] for t in window]
        free_before=sum(int(g["gap_ns"]) for g in parts)
        following=nxt["start"]-window[-1]["end"]
        time=nxt["start"]-prev["end"]
        assert time==sum(durations)+free_before+following
        marks=[e for e in markers if start<=ns(e["ts"])<end]
        if marks:
            assert len(marks)==len(cids)==1
            enq=deqs[next(iter(cids))]
            marker=marks[0]
            assert enq["tid"]==marker["tid"] and ns(marker["ts"])<=ns(enq["ts"]) and (
                ns(enq["ts"])+ns(enq["dur"]) <= ns(marker["ts"])+ns(marker["dur"]))
        result.append(dict(step=int(step),shape=[int(calls[window[0]["qid"]][k]) for k in ("batch","seq_len")],
                 before=name(prev),after=name(nxt),target_tasks=dict(Counter(t["name"] for t in window)),
                 target_work_slots=[window[0]["slot"],window[-1]["slot"]],number_of_target_qids=len(cids),
                 marker_us=[float(m["dur"]) for m in marks],wall_ms=time/1e6,copy_device_ms=sum(durations)/1e6,
                 copy_preceding_free_ms=free_before/1e6,copy_to_next_free_ms=following/1e6,
                 range_free_ms=(free_before+following)/1e6,
                 range_host_before=cpu_overlap(events,scope,prev["end"],window[0]["start"]) if marks else None,
                 range_host_after=cpu_overlap(events,scope,window[-1]["end"],nxt["start"]) if marks else None,
                 target_preceding_phase_overlap_ms={k:sum(int(g[k] or 0) for g in parts)/1e6 for k in
                 ("before_aten_ns","aten_before_workspace_ns","workspace_ns","aten_to_enqueue_remainder_ns",
                  "enqueue_ns","visible_queue_wait_ns","consumer_before_api_ns","acl_submission_api_ns","after_api_ns")}))
    return result


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("legs",type=Path)
    cli.add_argument("analysis",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    result={leg:digest(args.legs/leg,args.analysis/leg) for leg in ("01-A","02-B")}
    for a,b in zip(*result.values(),strict=True):
        for field in ("step","shape","before","after"):
            assert a[field]==b[field], (field,a[field],b[field])
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
