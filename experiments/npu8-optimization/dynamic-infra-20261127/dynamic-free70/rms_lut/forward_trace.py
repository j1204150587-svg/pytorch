"""Recompute per-RMS launch and hardware timings from individually drained Level1 windows."""
import argparse
from decimal import Decimal
import gzip
import json
from pathlib import Path
import statistics


def ns(value):
    return int(Decimal(str(value))*1000)


def span(row):
    t=ns(row["ts"])
    return t,t+ns(row["dur"])


def contains(outer,inner):
    return outer[0]<=inner[0] and inner[1]<=outer[1]


def union(spans):
    out=[]
    for a,b in sorted(spans):
        if a>=b:continue
        if out and a<=out[-1][1]:
            out[-1][1]=max(b,out[-1][1])
        else:
            out.append([a,b])
    return out


def measured_span(spans):
    return sum(b-a for a,b in union(spans))/1000


def quantiles(samples):
    if not samples:return None
    a=sorted(samples)
    return {"n":len(a),"p50":statistics.median(a),"p95":a[(95*len(a)+99)//100-1],
            "p99":a[(99*len(a)+99)//100-1],"max":a[-1]}


def real_work(row):
    t=row.get("args",{}).get("Task Type","")
    return t in {"AIC","AIV","AI_CORE","AICORE","AI_VECTOR_CORE","VECTOR_CORE","MIX_AIC",
        "MIX_AIV","MIX_AICORE","MIX_VECTOR_CORE","AI_CORE_MIX","MIX_AIC_AIV","MIX_AIV_AIC",
        "AI_CPU","AICPU"} or t.startswith(("KERNEL_","SDMA","HDMA","RDMA","HCCS_DMA","PCIE_DMA","MEMCPY","MEMSET"))


def collect(path):
    with (gzip.open if path.suffix==".gz" else open)(path,"rt",encoding="utf-8-sig") as f:
        z=json.load(f)
    allrows=z["traceEvents"] if isinstance(z,dict) else z
    pid,={r["pid"] for r in allrows if r.get("name")=="process_name" and r["args"]["name"]=="Ascend Hardware"}
    rows=[r for r in allrows if r.get("ph")=="X"]
    hardware_all=sorted((r for r in rows if r["pid"]==pid and "Task Id" in r.get("args",{})),key=span)
    hardware=[r for r in hardware_all if real_work(r)]
    markers={r["name"]:r for r in rows if r["name"].startswith("RMSLUT_")}
    windows=sorted((r for r in markers.values() if r["name"].endswith("_window")),key=span)
    result=[]
    for window in windows:
        name=window["name"].removesuffix("_window")
        _,route,shape,repeat=name.split("_")
        dispatch=markers[name+"_dispatch"]
        a,b=span(dispatch)
        w=span(window)
        local=[r for r in rows if contains(w,span(r))]
        hw=[r for r in hardware if contains(w,span(r))]
        if not hw:raise ValueError(f"missing hardware within synchronized window {name}")
        q=[r for r in local if r.get("cat")=="enqueue" and contains((a,b),span(r))]
        deq={r["args"]["correlation_id"]:r for r in local if r.get("cat")=="dequeue"}
        enqs=[]
        for enqueue in sorted(q,key=span):
            es,ee=span(enqueue)
            d=deq.get(enqueue["args"]["correlation_id"])
            record={"name":enqueue["name"],"start_from_dispatch_us":(es-a)/1000,"enqueue_us":(ee-es)/1000}
            if d:
                ds,de=span(d)
                children=[r for r in local if r["tid"]==d["tid"] and contains((ds,de),span(r))]
                record.update(queue_wait_us=(ds-ee)/1000,consumer_us=(de-ds)/1000,
                    node_launch_us=[float(r["dur"]) for r in children if r["name"]=="Node@launch"],
                    acl_launch_us=[float(r["dur"]) for r in children if r["name"].startswith("AscendCL@aclrtLaunch")])
            enqs.append(record)
        def taskrow(item):
            k,v=span(item)
            return {"name":item["name"],"stream":item["args"]["Physic Stream Id"],
                "task":item["args"]["Task Id"],"type":item["args"]["Task Type"],"duration_us":(v-k)/1000,
                "start_from_dispatch_us":(k-a)/1000}
        taskrows=[taskrow(item) for item in hw]
        other_hw=[taskrow(item) for item in hardware_all if contains(w,span(item)) and not real_work(item)]
        first,last=span(hw[0])[0],max(span(r)[1] for r in hw)
        earliest=min((span(r)[0] for r in q),default=None)
        before_tasks=measured_span([span(r) for r in local if r["name"].endswith("GetWorkspaceSize")
                                     and contains((a,b),span(r))])
        work=measured_span([span(r) for r in hw])
        result.append(dict(marker=name,route=route,shape=shape,repeat=repeat,dispatch_us=(b-a)/1000,
                pre_first_task_from_dispatch_us=(first-a)/1000,
                dispatch_to_first_enqueue_us=(earliest-a)/1000 if earliest is not None else None,
                workspace_union_us=before_tasks,device_work_union_us=work,
                device_interior_idle_us=(last-first)/1000-work,physical_tasks=taskrows,
                other_hardware_records=other_hw,queues=enqs))
    if not result:raise ValueError("no RMSLUT windows")
    groups={}
    for shape in sorted({r["shape"] for r in result}):
        for route in sorted({r["route"] for r in result}):
            selected=[r for r in result if (r["shape"],r["route"])==(shape,route)]
            if not selected:continue
            qrows=[q for r in selected for q in r["queues"]]
            groups[f"{shape}/{route}"]={k:quantiles([r[k] for r in selected if r[k] is not None])
                       for k in ("dispatch_us","dispatch_to_first_enqueue_us","pre_first_task_from_dispatch_us",
                            "device_work_union_us","device_interior_idle_us","workspace_union_us")}
            groups[f"{shape}/{route}"].update(physical_counts=[len(r["physical_tasks"]) for r in selected],
                all_hardware_counts=[len(r["physical_tasks"])+len(r["other_hardware_records"]) for r in selected],
                queue_wait_us=quantiles([r["queue_wait_us"] for r in qrows if "queue_wait_us" in r]),
                consumer_us=quantiles([r["consumer_us"] for r in qrows if "consumer_us" in r]),
                acl_launch_us=quantiles([d for r in qrows for d in r.get("acl_launch_us",[])]),
                node_launch_us=quantiles([d for r in qrows for d in r.get("node_launch_us",[])]))
    pairs=[]
    for shape in sorted({r["shape"] for r in result}):
        for repeat in sorted({r["repeat"] for r in result}):
            s=next((r for r in result if (r["shape"],r["repeat"],r["route"])==(shape,repeat,"S")),None)
            p=next((r for r in result if (r["shape"],r["repeat"],r["route"])==(shape,repeat,"P")),None)
            if not s or not p:continue
            key=lambda r:[(t["name"],t["type"],t["stream"]) for t in r["physical_tasks"]]
            pairs.append(dict(shape=shape,repeat=repeat,identical_stream_name_type_sequence=key(s)==key(p),
                         stock=key(s),plan=key(p)))
    return dict(trace=str(path),windows=result,per_shape=groups,same_kernel_pairs=pairs)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.write_text(json.dumps(collect(a.trace),indent=2)+"\n")
