"""Count exact RMS scopes and linked physical work in one or both training legs."""
import argparse
from collections import Counter,defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"causal_tril"))
from core_budget import intersect,merge,ns
from forward_trace import real_work


def analyze(trace):
    with (gzip.open if trace.suffix==".gz" else open)(trace,"rt",encoding="utf-8-sig") as f:
        doc=json.load(f)
    rows=doc if isinstance(doc,list) else doc["traceEvents"]
    hardware,={r["pid"] for r in rows if r.get("name")=="process_name" and r["args"]["name"]=="Ascend Hardware"}
    span=lambda r:(ns(r["ts"]),ns(r["ts"])+ns(r["dur"]))
    ev=[r for r in rows if r.get("ph")=="X"]
    physical=defaultdict(list)
    for r in ev:
        if r["pid"]==hardware:
            physical[r["tid"],span(r)[0]].append(r)
    edges=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r.get("cat")=="async_npu":edges[r["id"]][r["ph"]].append(r)
    links=[]
    for e in edges.values():
        if len(e["s"])!=1 or len(e["f"])!=1:continue
        a,b=e["s"][0],e["f"][0]
        hw=physical[b["tid"],ns(b["ts"])]
        if b["pid"]==hardware and len(hw)==1:
            links.append((a["tid"],ns(a["ts"]),hw[0]))
    free=merge(span(r) for r in ev if r["name"]=="Free")
    output=[]
    for step in sorted((r for r in ev if r["name"].startswith("ProfilerStep#")),key=span):
        beg,end=span(step)
        cur=[r for r in ev if beg<=span(r)[0] and span(r)[1]<=end]
        all_work=sorted((r for r in cur if r["pid"]==hardware and real_work(r)),key=span)
        kernels=[r for r in all_work if r["name"]=="lut_forward"]
        def task_info(t):
            return {"name":t["name"],"stream":t["args"].get("Physic Stream Id"),"type":t["args"].get("Task Type"),
                    "task_id":t["args"].get("Task Id"),"kernel_us":float(t["dur"])}
        phase={}
        for name in ("hstu_forward_and_loss","deepspeed_backward","optimizer_step_including_native_deepspeed_norm","input_generation_and_h2d"):
            r,=[r for r in cur if r["name"]==name]
            a,b=span(r)
            phase[name]={"wall_ms":(b-a)/1e6,"free_overlap_ms":intersect([(a,b)],free)/1e6}
        def scopes(name):
            found=sorted((r for r in cur if r["name"]==name),key=span)
            if len(found)!=9:
                options=sorted({r["name"] for r in cur if "RMS" in r["name"] or "Rms" in r["name"]})
                raise ValueError(f"expected 9 {name} in {step['name']}, got {len(found)}; RMS labels {options}")
            scoped=[]
            direct=[]
            for i,r in enumerate(found):
                a,b=span(r)
                tasks=sorted((dev for tid,t,dev in links if tid==r["tid"] and a<=t<b),key=span)
                row={"wall_us":(b-a)/1e3,"free_overlap_us":intersect([(a,b)],free)/1e3,
                     "async_linked_tasks":[task_info(t) for t in tasks]}
                if kernels and name=="SharedRMS":
                    if len(kernels)!=len(found) or len({k["args"]["Physic Stream Id"] for k in kernels})!=1:
                        raise ValueError("unable to reconcile nine ordered LUT tasks on one stream")
                    enqueue,=[e for e in cur if e["name"]=="Enqueue@lut_forward" and e["tid"]==r["tid"] and a<=span(e)[0]<b]
                    deq,=[e for e in cur if e["name"]=="Dequeue@lut_forward" and
                          e["args"]["correlation_id"]==enqueue["args"]["correlation_id"]]
                    row["queue_id"]=enqueue["args"]["correlation_id"]
                    row["owner_to_enqueue_us"]=(span(enqueue)[0]-a)/1e3
                    row["consumer_us"]=float(deq["dur"])
                    row["physical_task_in_stream_order"]=task_info(kernels[i])
                    direct.append(row["physical_task_in_stream_order"])
                scoped.append(row)
            byflow=[t for z in scoped for t in z["async_linked_tasks"]]
            tasks=direct or byflow
            return dict(occurrences=scoped,linked_devices=len(byflow),verified_device_tasks=len(tasks),
                        linkage="same-stream FIFO plus matching enq/deq" if direct else "async_npu",
                        device_names=dict(Counter(t["name"] for t in tasks)),
                        phys_stream=dict(Counter(t["stream"] for t in tasks)),
                        wall_ms=sum(z["wall_us"] for z in scoped)/1000,
                        free_intersection_ms=sum(z["free_overlap_us"] for z in scoped)/1000)
        output.append(dict(step=step["name"],device_work_count=len(all_work),by_stream=dict(Counter(r["args"]["Physic Stream Id"] for r in all_work)),
                      phases=phase,forward=scopes("SharedRMS"),
                      backward=scopes("autograd::engine::evaluate_function: SharedRMSBackward")))
    if not output:raise ValueError("no training steps")
    return {"trace_sha256":hashlib.sha256(trace.read_bytes()).hexdigest(),"steps":output}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.write_text(json.dumps(analyze(a.trace),indent=2)+"\n")
