"""Summarize real launches, equal-device-work Level1 markers, and two types of paired timing."""
import argparse
from collections import Counter
from decimal import Decimal
import gzip
import json
import math
from pathlib import Path
import statistics


def stats(values):
    v=sorted(values)
    return dict(n=len(v),p50=statistics.median(v),p95=v[math.ceil(.95*len(v))-1],p99=v[math.ceil(.99*len(v))-1],maximum=v[-1])


def profile(trace):
    opener=gzip.open if trace.suffix==".gz" else open
    with opener(trace,"rt") as f:data=json.load(f)
    ev=data if isinstance(data,list) else data["traceEvents"]
    pids={e["args"]["name"]:e["pid"] for e in ev if e["name"]=="process_name"}
    rows=[]
    for e in ev:
        if e.get("ph")=="X":
            left=int(Decimal(str(e["ts"]))*1000)
            rows.append(dict(e,l=left,r=left+int(Decimal(str(e["dur"]))*1000)))
    marks=sorted((e for e in rows if e["name"].startswith("Pending_")),key=lambda e:e["l"])
    reports=[]
    for m in marks:
        own=[e for e in rows if m["l"]<=e["l"] and e["r"]<=m["r"]]
        hardware=[e for e in own if e["pid"]==pids["Ascend Hardware"] and e["name"]=="aclnnAdd_AddAiCore_Add"]
        actual=[e for e in own if e["pid"]==pids["CANN"] and e["name"].startswith("AscendCL@aclrtLaunchKernel")]
        real=Counter(e["name"] for e in own if e["pid"]==pids["Ascend Hardware"] and
                     (e.get("args",{}).get("Task Type") in {"AI_VECTOR_CORE","AI_CORE","AICORE","MIX_AIC","MIX_AIV"} or
                      str(e.get("args",{}).get("Task Type","")).startswith(("SDMA","HDMA","MEMCPY","RDMA","KERNEL_"))))
        st={e["args"]["Physic Stream Id"] for e in hardware}
        if len(actual)!=32 or len(hardware)!=32 or len(st)!=1 or real!={"aclnnAdd_AddAiCore_Add":32}:
            raise ValueError((m["name"],len(actual),len(hardware),st,real))
        reports.append(dict(marker=m["name"],hardware_tasks=real,stream=st.pop(),real_acl_calls=len(actual),
                            host_launch_us=stats([(x["r"]-x["l"])/1000 for x in actual])))
    if len(reports)!=4 or len({x["stream"] for x in reports})!=1:raise ValueError(reports)
    return reports


def main(root):
    dump=lambda p:json.loads(p.read_bytes())
    gate=dump(root/"micro.json")
    if not gate.get("complete"):raise ValueError("Micro is incomplete")
    process={}
    cross={}
    for i,arm in ((3,"A"),(4,"B"),(5,"B"),(6,"A")):
        raw=dump(root/f"part-{i}-{arm}.json")
        tag=f"{i}:{arm}"
        process[tag]={}
        for k,x in raw["windows"].items():
            w=x["no_call_timing"]
            process[tag][k]={kind:stats([r[kind]/r["n"]/1000 for r in w]) for kind in ("host_ns","complete_ns")}
            process[tag][k]["single_us"]=stats([v/1000 for r in x["individual"] for v in r["launch_ns"]])
            process[tag][k]["cpu_migrations_observed"]=sum(r["cpu_before"]!=r["cpu_after"] for r in w)
    for k in next(iter(process.values())):
        cross[k]={order:{m:process[b][k][m]["p50"]-process[a][k][m]["p50"] for m in ("host_ns","complete_ns","single_us")}
                  for order,a,b in (("AB","3:A","4:B"),("BA","6:A","5:B"))}
    paired={}
    for k,rows in dump(root/"part-7-paired.json")["paired"].items():
        paired[k]={}
        for order in ("AB","BA"):
            matched=[r for r in rows if r["order"]==order]
            paired[k][order]={}
            for m in ("host_ns","complete_ns"):
                v=[(r["B"][m]-r["A"][m])/r["A"]["n"]/1000 for r in matched]
                stable=[(r["B"][m]-r["A"][m])/r["A"]["n"]/1000 for r in matched
                        if len({r[arm][pos] for arm in "AB" for pos in ("cpu_before","cpu_after")})==1]
                paired[k][order][m]=dict(**stats(v),wins=sum(t<0 for t in v),delta_us=v,stable_cpu=stats(stable) if stable else {})
    traces=list((root/"profile").rglob("trace_view.json"))
    if len(traces)!=1:raise ValueError(traces)
    out=dict(durations_unit="us/op; *_ns labels identify source totals in ns before normalization",
             process=process,process_B_minus_A=cross,same_process_pair_B_minus_A=paired,profile=profile(traces[0]))
    (root/"analysis.json").write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("dir",type=Path)
    main(a.parse_args().dir)
