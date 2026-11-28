"""Compute cross-process AB/BA launch timings and verify the Level1 work signatures."""
from collections import Counter
from decimal import Decimal
import json
import math
from pathlib import Path
import statistics
import sys


def percentiles(series):
    v=sorted(series)
    return dict(n=len(v),p50=statistics.median(v),p95=v[math.ceil(.95*len(v))-1],
                p99=v[math.ceil(.99*len(v))-1],maximum=v[-1])


def profile(path):
    source=json.loads(path.read_bytes())
    events=source if isinstance(source,list) else source["traceEvents"]
    pids={e["args"]["name"]:e["pid"] for e in events if e.get("name")=="process_name"}
    rows=[]
    for e in events:
        if e.get("ph")=="X":
            start=int(Decimal(str(e["ts"]))*1000)
            end=start+int(Decimal(str(e["dur"]))*1000)
            rows.append((start,end,e))
    markers={}
    for ls,rs,mark in rows:
        if not mark["name"].startswith("StreamLayout_"): continue
        inside=[(l,r,e) for l,r,e in rows if ls<=l and r<=rs]
        launches=[(r-l)/1000 for l,r,e in inside if e["pid"]==pids["CANN"] and
                  e["name"].startswith("AscendCL@aclrtLaunchKernel")]
        tasks=[e for _,_,e in inside if e["pid"]==pids["Ascend Hardware"] and (
               e.get("args",{}).get("Task Type") in {"AI_VECTOR_CORE","AI_CORE","AICORE","MIX_AIC","MIX_AIV"}
               or str(e.get("args",{}).get("Task Type","")).startswith(("SDMA","HDMA","MEMCPY","RDMA","KERNEL_")))]
        assert len(launches)==len(tasks)==32
        types=Counter((e["name"],str(e["args"]["Physic Stream Id"])) for e in tasks)
        assert len(types)==1
        (kernel,stream),=types.keys()
        markers[mark["name"]]=dict(kernel=kernel,stream=stream,n_work=len(tasks),n_acl=len(launches),
                                   acl_us=percentiles(launches))
    assert len(markers)==2
    return markers


def main(root):
    summary=json.loads((root/"summary.json").read_bytes())
    assert len(summary)==4 and all(r["rc"]==0 for r in summary)
    parts=[]
    masks=[]
    for row in summary:
        raw=json.loads((root/row["result"]).read_bytes())
        assert raw["complete"] and len(raw["bitwise"])==60 and all(raw["bitwise"])
        assert raw["arm"]==row["arm"]
        masks.append(raw["shape_sequence"])
        test={}
        for name,v in raw["windows"].items():
            w=v["unprofiled"]
            assert len(w)==21
            test[name]={key:percentiles([t[key]/t["n"]/1000 for t in w]) for key in ("host_ns","complete_ns")}
            test[name]["single_us"]=percentiles([n/1000 for t in v["individual"] for n in t["launch_ns"]])
            test[name]["cpu_migrated_windows"]=sum(t["cpu_before"]!=t["cpu_after"] for t in w)
        parts.append(dict(arm=raw["arm"],split=raw["current_geometry"]==2,stream_mod128=raw["stream_object"]%128,
                          initial_counts=raw["counters_before"],after_counts=raw["counters_after"],
                          created_classes=[g for _,g in raw["created"]],shapes=len(raw["bitwise"]),measured=test))
    assert all(s==masks[0] for s in masks)
    delta={}
    for order,baseline,candidate in (("AB",0,1),("BA",3,2)):
        a=parts[baseline]["measured"]
        b=parts[candidate]["measured"]
        delta[order]={name:{metric:b[name][metric]["p50"]-a[name][metric]["p50"]
                            for metric in ("host_ns","complete_ns","single_us")} for name in a}
    reports={}
    for trace in root.glob("part-*-profile/**/trace_view.json"):
        reports.update(profile(trace))
    assert len(reports)==4
    signatures={(v["kernel"],v["stream"],v["n_work"],v["n_acl"]) for v in reports.values()}
    result=dict(unit="us per real Add; host_ns/complete_ns identify the source window totals",
                parts=parts,paired_B_minus_A=delta,level1=reports,same_physical_signature=len(signatures)==1)
    (root/"micro-results.json").write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__": main(Path(sys.argv[1]))
