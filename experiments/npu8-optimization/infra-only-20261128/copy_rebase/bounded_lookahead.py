"""Offline bounded consumer lookahead for the observed stream-44 eager copy segment."""
from argparse import ArgumentParser
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path
from statistics import median


def ns(t):
    return int(Decimal(str(t))*1000)


def load_csv(folder,name):
    with gzip.open(folder/(name+".csv.gz"),"rt",newline="") as src:return list(csv.DictReader(src))


def summarize_us(vals):
    vals=sorted(vals)
    return dict(count=len(vals),sum_ms=sum(vals)/1e6,
                p50_us=median(vals)/1000,p95_us=vals[int((len(vals)-1)*.95)]/1000,max_us=max(vals)/1000)


def model(group,stats,barriers,hold_us,policy):
    # Freeze the observed producer arrival times and consumer service starts. This is a window
    # opportunity model, not a counterfactual trace of the model or an estimate of training E2E.
    held,head_latency,collapsed,head_penalty,seen,publication_deltas=[],[],[],[],[],[]
    singleton_wait=[]
    cursor=0
    while cursor<len(group):
        head=stats[group[cursor]]
        ready=max(head["deq_start"],head["published"])
        deadline=max(ready,head["published"]+1000*hold_us)
        # Any barrier published during the open interval closes the group at once. Already queued
        # older work has been dispatched before this head. A noncopy resolving at the exact
        # publication timestamp wins the tie.
        bidx=bisect_right(barriers,head["published"])
        barrier=barriers[bidx] if bidx<len(barriers) else 2**63-1
        flush=max(ready,min(deadline,barrier))
        next_pub=stats[group[cursor+1]]["published"] if cursor+1<len(group) else 2**63-1
        if policy=="first_partner" and next_pub<=deadline and next_pub<barrier:
            flush=max(ready,next_pub)
        end=cursor+1
        while end<len(group) and stats[group[end]]["published"]<=flush and stats[group[end]]["published"]<barrier:
            end+=1
        keys=group[cursor:end]
        # At the end of a consecutive segment, the first future barrier flushes pending work.
        # There is no clairvoyant release as soon as the final copy in the group arrives.
        wrap=max(0,flush-head["deq_start"])
        held.append(max(0,flush-ready))
        head_latency.append(flush-head["published"])
        if len(keys)==1: singleton_wait.append(max(0,flush-ready))
        collapsed.extend(keys[1:])
        seen.append(len(keys))
        gap=head.get("gap")
        if gap:
            # Hardware start minus the original ACL return. Only the part of added head hold
            # outside that observed interval could move that first device start in this model.
            head_penalty.append(max(0,wrap-head["actual_launch_to_hardware"]))
        if end<len(group):publication_deltas.append(stats[group[end]]["published"]-head["published"])
        cursor=end
    return dict(heads=len(seen),collapsed=len(collapsed),group_sizes=dict(Counter(seen)),
                additional_hold=summarize_us(held),publish_to_flush=summarize_us(head_latency),
                singleton_additional_hold_ms=sum(singleton_wait)/1e6,
                linked_gap_heads=len(head_penalty),estimated_uncovered_head_delay_ms=sum(head_penalty)/1e6,
                removed_old_service_ms={name:sum(stats[k][name] for k in collapsed)/1e6
                    for name in ("dequeue_ns","aclnn_ns","acl_memcpy_ns")},
                removed_old_gap_intersection_ms={name:sum(stats[k].get(name,0) for k in collapsed)/1e6
                    for name in ("gap","consumer_before_api_gap","acl_gap","after_api_gap")})


def main():
    p=ArgumentParser()
    p.add_argument("analysis",type=Path)
    p.add_argument("trace",type=Path)
    p.add_argument("address_summary",type=Path)
    p.add_argument("out",type=Path)
    cfg=p.parse_args()
    address=json.loads(cfg.address_summary.read_bytes())["stream_address_census"]["44"]["steps_from_identical_range"]
    assert all(r["other_calls"]==1052 and r["other_adjacency"]["both_forward"]==1051 for r in address)
    with gzip.open(cfg.trace,"rt",encoding="utf-8-sig") as src:
        trace=json.load(src)
    events=trace if isinstance(trace,list) else trace["traceEvents"]
    en,deq,lanes,api,markers={},{},defaultdict(list),defaultdict(list),[]
    hardware=defaultdict(list)
    steps={}
    for e in events:
        if e.get("ph")!="X":continue
        cat,name=e.get("cat"),e["name"]
        start=ns(e["ts"])
        end=start+ns(e["dur"])
        if name.startswith("ProfilerStep#"):
            steps[name.split("#")[-1]]=(start,end)
        elif name=="MEMCPY_ASYNC" and str(e.get("args",{}).get("Physic Stream Id"))=="44":
            hardware[str(e["args"]["Task Id"])].append(start)
        elif cat in ("enqueue","dequeue"):
            cid=str(e["args"]["correlation_id"])
            if cat=="enqueue":en[cid]=end
            else:
                deq[cid]=(e["tid"],start,end)
                lanes[e["tid"]].append((start,cid))
        elif name in ("AscendCL@aclnnInplaceCopy","AscendCL@aclrtMemcpyAsync"):
            api[e["tid"]].append((start,end,name))
        elif name.startswith("AscendCL@aclrt") and any(t in name for t in ("Synchronize","WaitEvent","RecordEvent")):
            markers.append(start)
    del events,trace
    for rows in api.values():rows.sort()
    for rows in lanes.values():rows.sort()
    api_starts={tid:[r[0] for r in rows] for tid,rows in api.items()}
    calls={(r["step"],r["queue_id"]):r for r in load_csv(cfg.analysis,"queue_calls")}
    gaps={(r["step"],r["queue_id"]):r for r in load_csv(cfg.analysis,"device_gaps") if r["state"]=="linked"}
    segments=defaultdict(list)
    task_ids={}
    for r in load_csv(cfg.analysis,"work_tasks"):
        key=(r["step"],r["queue_id"])
        if r["stream"]=="44" and r["task"]=="MEMCPY_ASYNC" and key in calls and calls[key]["op"]=="aclnnInplaceCopy":
            segments[key[0]].append((int(r["stream_work_slot"]),key[1]))
            task_ids[key]=r["task_id"]
    out={}
    for step,s in sorted(segments.items()):
        s.sort()
        group=[cid for _,cid in s]
        assert len(group)==1052 and all(en[a]<=en[b] for a,b in zip(group,group[1:]))
        member=set(group)
        tids={deq[cid][0] for cid in group}
        assert len(tids)==1
        tid=tids.pop()
        lane=[cid for _,cid in lanes[tid]]
        first=lane.index(group[0])
        assert lane[first:first+len(group)]==group
        stats={}
        for cid in group:
            _,lo,hi=deq[cid]
            base=dict(deq_start=lo,published=en[cid],dequeue_ns=hi-lo,aclnn_ns=0,acl_memcpy_ns=0)
            counts=Counter()
            for a,b,name in api[tid][bisect_left(api_starts[tid],lo):]:
                if a>=hi:break
                if b<=hi:
                    counts[name]+=1
                    attr="aclnn_ns" if name=="AscendCL@aclnnInplaceCopy" else "acl_memcpy_ns"
                    base[attr]+=b-a
                    if name=="AscendCL@aclrtMemcpyAsync":base["acl_end"]=b
            assert sorted(counts.values())==[1,1], (cid,counts)
            hw,=[t for t in hardware[task_ids[step,cid]] if steps[step][0]<=t<steps[step][1]]
            base["actual_launch_to_hardware"]=max(0,hw-base["acl_end"])
            g=gaps.get((step,cid))
            if g:
                base.update(gap=int(g["gap_ns"]),consumer_before_api_gap=int(g["consumer_before_api_ns"]),
                            acl_gap=int(g["acl_submission_api_ns"]),after_api_gap=int(g["after_api_ns"]))
            stats[cid]=base
        lane_barriers=sorted(en[cid] for cid in lane if cid not in member and cid in en)
        global_barriers=sorted([time for cid,time in en.items() if cid not in member]+markers)
        out[step]=dict(shape=[int(calls[step,group[0]]["batch"]),int(calls[step,group[0]]["seq_len"])],
            physical_stream=44,physical_submissions=len(group),
            earlier_than_enqueue_end=sum(deq[cid][1]<en[cid] for cid in group),
            consecutive_publication=summarize_us([en[b]-en[a] for a,b in zip(group,group[1:])]),
            original_service_ms={name:sum(item[name] for item in stats.values())/1e6 for name in ("dequeue_ns","aclnn_ns","acl_memcpy_ns")},
            windows={scope:{policy:{str(cap):model(group,stats,barr,cap,policy) for cap in (0,1,2,5,10,20,50)}
                      for policy in ("full_window","first_partner")} for scope,barr in
                      (("consumer_lane_barriers",lane_barriers),("all_visible_barriers",global_barriers))})
    cfg.out.write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
