"""Match Cast→LpNorm requests and count visible non-delaying lookahead inside unmodified dynamic HSTU."""
import argparse
from bisect import bisect_left,bisect_right
from collections import Counter,defaultdict
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path


CAST="aclnnInplaceCopy_CastAiCore_Cast"
NORM="aclnnLinalgVectorNorm_LpNormV2_LpNormV2"


def ns(x):return int(Decimal(str(x))*1000)


def read(folder,name):
    with gzip.open(folder/(name+".csv.gz"),"rt",newline="") as f:return list(csv.DictReader(f))


def dist(items,scale=1):
    vals=sorted(items)
    return {"n":len(vals),**{name:vals[int((len(vals)-1)*fn)]/scale for name,fn in
        (("min",0),("p50",.5),("p95",.95),("p99",.99),("max",1))}} if vals else {"n":0}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("analysis",type=Path)
    p.add_argument("trace",type=Path)
    p.add_argument("out",type=Path)
    cfg=p.parse_args()
    cfg.out.mkdir(exist_ok=True,parents=True)
    with gzip.open(cfg.trace,"rt",encoding="utf-8-sig") as f: trace=json.load(f)
    ev=trace if isinstance(trace,list) else trace["traceEvents"]
    en,de,step_bounds={},{},{}
    lane,api,hw=defaultdict(list),defaultdict(list),defaultdict(list)
    barriers=[]
    for e in ev:
        if e.get("ph")!="X":continue
        cat,name=e.get("cat"),e["name"]
        t=ns(e["ts"]);hi=t+ns(e["dur"]);tid=e["tid"];attrs=e.get("args",{})
        if cat in ("enqueue","dequeue"):
            cid=str(attrs["correlation_id"]); dest=en if cat=="enqueue" else de
            assert cid not in dest
            dest[cid]=(tid,t,hi)
            if cat=="dequeue":lane[tid].append((t,cid))
        elif name.startswith("ProfilerStep#"):step_bounds[name.split("#")[-1]]=(t,hi)
        elif name in ("AscendCL@aclnnInplaceCopy","AscendCL@aclnnLinalgVectorNorm","AscendCL@aclrtLaunchKernelWithHostArgs"):
            api[tid].append((t,hi,name))
        elif name in (CAST,NORM) and "Physic Stream Id" in attrs:
            hw[str(attrs["Physic Stream Id"]),str(attrs["Task Id"]),name].append((t,hi))
        elif name.startswith("AscendCL@aclrt") and any(token in name for token in ("Synchronize","WaitEvent","RecordEvent","QueryEvent","QueryStream")):
            barriers.append(t)
    del ev,trace
    for rs in lane.values():rs.sort()
    for rs in api.values():rs.sort()
    api_start={tid:[r[0] for r in rs] for tid,rs in api.items()}
    positions={cid:(t,i) for t,rs in lane.items() for i,(_,cid) in enumerate(rs)}
    q={(r["step"],r["queue_id"]):r for r in read(cfg.analysis,"queue_calls")}
    work=defaultdict(list);stream_work=defaultdict(list);gaps={}
    for r in read(cfg.analysis,"work_tasks"):
        work[r["step"],r["queue_id"]].append(r)
        stream_work[r["step"],r["stream"]].append(r)
    for r in read(cfg.analysis,"device_gaps"):
        if r["state"]=="linked":gaps[r["step"],r["queue_id"],r["task_id"]]=r
    result={};all_rows=[]
    for step in ("4","5"):
        tasks=sorted(stream_work[step,"47"],key=lambda r:int(r["stream_work_slot"]))
        pairs=[(a,b) for a,b in zip(tasks,tasks[1:]) if a["task"]==CAST and b["task"]==NORM]
        assert len(pairs)==1052
        ids=[r["queue_id"] for pair in pairs for r in pair]
        member_ids=set(ids)
        assert len(member_ids)==2104
        tids={de[k][0] for k in ids};producers={en[k][0] for k in ids}
        tid,=tids;producer,=producers
        laneids=[k for _,k in lane[tid]]
        index=positions[ids[0]][1]
        assert laneids[index:index+len(ids)]==ids
        others_start=sorted(v[1] for k,v in en.items() if k not in member_ids)
        others_end=sorted(v[2] for k,v in en.items() if k not in member_ids)
        lane_cids=[k for _,k in lane[tid] if k in en]
        pub_bounds={mode:sorted(en[k][i] for k in lane_cids) for mode,i in (("start",1),("end",2))}
        closed=sorted(de[k][2] for k in lane_cids)
        stats={}
        for (a,b) in pairs:
            for r in (a,b):
                cid=r["queue_id"];call=q[step,cid]
                assert len(work[step,cid])==1 and call["role"]=="main" and not call["workspace_ns"]
                assert call["acl_api_sequence"]=="AscendCL@aclrtLaunchKernelWithHostArgs"
                d=de[cid];canned=[]
                for t,hi,name in api[tid][bisect_left(api_start[tid],d[1]):]:
                    if t>=d[2]:break
                    if hi<=d[2]:canned.append((name,t,hi))
                core,=[z for z in canned if z[0].startswith("AscendCL@aclnn")]
                launch,=[z for z in canned if z[0]=="AscendCL@aclrtLaunchKernelWithHostArgs"]
                physical,=[z for z in hw["47",r["task_id"],r["task"]] if step_bounds[step][0]<=z[0]<step_bounds[step][1]]
                gap=gaps.get((step,cid,r["task_id"]),{})
                stats[cid]=dict(dequeue=d[1],dequeue_end=d[2],aclnn=core[1],aclnn_end=core[2],
                    acl=launch[1],acl_end=launch[2],hardware=physical[0],hardware_end=physical[1],
                    gap=int(gap.get("gap_ns") or 0),acl_gap=int(gap.get("acl_submission_api_ns") or 0),
                    producer_gap=int(gap.get("aten_to_enqueue_remainder_ns") or 0),qgap=int(gap.get("visible_queue_wait_ns") or 0),
                    linked=bool(gap),slot=r["stream_work_slot"])
        status={"shape":[int(q[step,ids[0]]["batch"]),int(q[step,ids[0]]["seq_len"])],"pairs":len(pairs),
            "same_producer_tid":producer,"same_consumer_tid":tid,"physical_stream":47,
            "cast_gap_ms":sum(stats[k]["gap"] for k in ids[::2])/1e6,"norm_gap_ms":sum(stats[k]["gap"] for k in ids[1::2])/1e6}
        block_lo=en[ids[0]][1];block_hi=de[ids[-1]][2]
        status["visible_barriers_during_segment"]=dict(other_enqueue_starts=bisect_left(others_start,block_hi)-bisect_right(others_start,block_lo),
            other_enqueue_ends=bisect_left(others_end,block_hi)-bisect_right(others_end,block_lo),
            acl_sync_event_queries=sum(block_lo<=t<=block_hi for t in barriers))
        rows=[]
        for i,(a,b) in enumerate(pairs):
            cast=a["queue_id"];norm=b["queue_id"];nc=ids[2*i+2] if 2*i+2<len(ids) else None;nn=ids[2*i+3] if nc else None
            head=stats[cast]
            row=dict(step=step,pair=i,stream=47,cast_queue=cast,norm_queue=norm,cast_slot=head["slot"],norm_slot=stats[norm]["slot"],
                cast_gap_ns=head["gap"],norm_gap_ns=stats[norm]["gap"],cast_acl_gap_ns=head["acl_gap"],norm_acl_gap_ns=stats[norm]["acl_gap"],
                cast_aclnn_ns=head["aclnn_end"]-head["aclnn"],cast_acl_ns=head["acl_end"]-head["acl"],
                norm_aclnn_ns=stats[norm]["aclnn_end"]-stats[norm]["aclnn"],norm_acl_ns=stats[norm]["acl_end"]-stats[norm]["acl"])
            for name in ("dequeue","aclnn","acl","acl_end","dequeue_end","hardware","hardware_end"):
                mark=head[name]
                for mode,idx in (("start",1),("end",2)):
                    row[f"{name}_{mode}_inflight"]=bisect_right(pub_bounds[mode],mark)-bisect_right(closed,mark)
                    for kind,member in (("same_norm",norm),("next_cast",nc),("next_norm",nn)):
                        if not member:continue
                        row[f"{name}_{mode}_{kind}_lead_ns"]=mark-en[member][idx]
            rows.append(row)
        summary={}
        for mark in ("dequeue","aclnn","acl","acl_end","dequeue_end","hardware","hardware_end"):
            for mode in ("start","end"):
                key=f"{mark}_{mode}";t={}
                t["inflight_host_not_yet_closed"]=dist([r[key+"_inflight"] for r in rows])
                for kind in ("same_norm","next_cast","next_norm"):
                    col=key+"_"+kind+"_lead_ns";valid=[r for r in rows if col in r]
                    yes=[r for r in valid if r[col]>=0]
                    t[kind]=dict(n=len(valid),ready=len(yes),lead_us=dist([r[col] for r in valid],1000),
                        wait_for_publication_us=dist([max(0,-r[col]) for r in valid],1000),
                        eligible_cast_gap_ms=sum(r["cast_gap_ns"] for r in yes)/1e6,
                        eligible_norm_gap_ms=sum(r["norm_gap_ns"] for r in yes)/1e6,
                        eligible_pair_acl_gap_ms=sum(r["cast_acl_gap_ns"]+r["norm_acl_gap_ns"] for r in yes)/1e6,
                        eligible_cast_acl_ms=sum(r["cast_acl_ns"] for r in yes)/1e6,
                        eligible_norm_acl_ms=sum(r["norm_acl_ns"] for r in yes)/1e6,
                        eligible_max_of_one_unchanged_acl_ms=sum(max(r["cast_acl_ns"],r["norm_acl_ns"]) for r in yes)/1e6)
                summary[key]=t
        status["anchors"]=summary
        status["dequeue_order_contiguous"]=True
        result[step]=status;all_rows.extend(rows)
    (cfg.out/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    cols=list(dict.fromkeys(k for r in all_rows for k in r))
    with gzip.open(cfg.out/"pairs.csv.gz","wt",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader();writer.writerows(all_rows)


if __name__=="__main__":
    main()
