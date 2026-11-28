"""Join rank-1 scalar HCCL Enqueue/Dequeue, CANN, communication, SDMA and official Free."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from decimal import Decimal
import gzip
import json
from pathlib import Path


def ns(value): return int(Decimal(str(value))*1000)


def overlap(span, intervals):
    l,r=span
    if l>=r:return 0
    return sum(max(0,min(r,b)-max(l,a)) for a,b in intervals if b>l and a<r)


def union(intervals):
    result=[]
    for l,r in sorted(intervals):
        if result and l<=result[-1][1]: result[-1]=(result[-1][0],max(result[-1][1],r))
        else: result.append((l,r))
    return result


def collect(path):
    with gzip.open(path,"rt",encoding="utf8") as fp:raw=json.load(fp)
    ev=raw if isinstance(raw,list) else raw["traceEvents"]
    pids={e["args"]["name"]:e["pid"] for e in ev if e["name"]=="process_name"}
    data=[]
    for e in ev:
        if e.get("ph")!="X":continue
        s=ns(e["ts"])
        data.append((s,s+ns(e["dur"]),e))
    free=[(a,b) for a,b,e in data if e["pid"]==pids["Overlap Analysis"] and e["name"]=="Free"]
    official_comm=union((a,b) for a,b,e in data if e["pid"]==pids["Overlap Analysis"] and e["name"]=="Communication")
    compute=union((a,b) for a,b,e in data if e["pid"]==pids["Overlap Analysis"] and e["name"]=="Computing")
    hardware=[(a,b,e) for a,b,e in data if e["pid"]==pids["Ascend Hardware"] and
        e["args"].get("Task Type","").startswith(("AI_","MIX_","SDMA","HDMA","RDMA"))]
    comm=[r for r in data if r[2]["pid"]==pids["Communication"] and r[2]["name"].startswith("hcom_allReduce__")]
    enq={r[2]["args"]["correlation_id"]:r for r in data if r[2]["name"]=="Enqueue@HcclAllreduce"}
    dqs={r[2]["args"]["correlation_id"]:r for r in data if r[2]["name"]=="Dequeue@HcclAllreduce"}
    free=union(free)
    reports=[]
    for step in sorted((v for v in data if v[2]["name"].startswith("ProfilerStep#")),key=lambda v:v[0]):
        rows=[]
        for cid,(ea,eb,ee) in sorted(enq.items(),key=lambda x:x[1][0]):
            if not step[0]<=ea<step[1]:continue
            da,db,de=dqs[cid]
            def same_deq(v):return v[2]["tid"]==de["tid"] and da<=v[0] and v[1]<=db
            ac,=[x for x in data if x[2]["name"]=="AscendCL@hcom_allReduce_" and same_deq(x)]
            launch,=[x for x in data if x[2]["name"]=="Node@launch" and same_deq(x) and x[:2]==ac[:2]]
            conn=launch[2]["args"]["connection_id"]
            dev,=[x for x in comm if x[2]["args"].get("connection_id")==conn]
            meta=dev[2]["args"]
            if int(meta["count"])!=1 or int(meta["rank_size"])!=1:continue
            tasks=sorted((t for t in hardware if t[2]["args"].get("connection_id")==conn), key=lambda t:t[0])
            task_spans=union(t[:2] for t in tasks)
            par,=[t for t in data if t[2]["name"]=="c10d::allreduce_" and t[2]["tid"]==ee["tid"] and t[0]<=ea and eb<=t[1]]
            before=max((t for t in hardware if t[1]<=tasks[0][0]),key=lambda t:t[1])
            nxt=min((t for t in hardware if t[0]>=tasks[-1][1]),key=lambda t:t[0])
            prior=(before[1],tasks[0][0]);after=(tasks[-1][1],nxt[0])
            phases=dict(producer_to_enqueue=(par[0],ea),enqueue=(ea,eb),enqueue_to_consumer=(eb,da),
                        consumer=(da,db),host_hccl=ac[:2],device_comm=dev[:2],previous_work_to_first=prior,last_to_next=after)
            sheet={k:dict(span_us=(b-a)/1000,official_free_us=overlap((a,b),free)/1000,
                          official_compute_us=overlap((a,b),compute)/1000,official_comm_us=overlap((a,b),official_comm)/1000)
                   for k,(a,b) in phases.items()}
            nested=[(a,b) for a,b,e in data if e["name"]=="AscendCL@aclrtReduceAsync" and same_deq((a,b,e))]
            rows.append(dict(qid=cid,connection_id=conn,dtype=meta["data_type"],count=meta["count"],
                physical=[(t[2]["args"]["Physic Stream Id"],t[2]["args"]["Task Id"],t[2]["name"],(t[1]-t[0])/1000) for t in tasks],
                prior_task=[before[2]["name"],before[2]["args"]["Physic Stream Id"]],next_task=[nxt[2]["name"],nxt[2]["args"]["Physic Stream Id"]],
                host_reduce_async_calls=len(nested),host_reduce_async_sum_us=sum(b-a for a,b in nested)/1000,
                host_reduce_async_free_us=sum(overlap(x,free) for x in nested)/1000,
                hccl_to_next_device_free_us=overlap((max(ac[0],after[0]),min(ac[1],after[1])),free)/1000,
                hccl_before_first_device_free_us=overlap((max(ac[0],prior[0]),min(ac[1],prior[1])),free)/1000,
                hccl_same_step_free_interval=[ac[0],ac[1]],phases=sheet))
        reports.append(dict(step=int(step[2]["name"].split("#")[-1]),rows=rows,
            host_hccl_free_union_us=sum(overlap(span,free) for span in union(r["hccl_same_step_free_interval"] for r in rows))/1000))
    return reports


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("out",type=Path)
    cli.add_argument("inputs",nargs="+",type=Path)
    args=cli.parse_args()
    args.out.write_text(json.dumps({str(p):collect(p) for p in args.inputs},indent=2)+"\n",encoding="utf8")
