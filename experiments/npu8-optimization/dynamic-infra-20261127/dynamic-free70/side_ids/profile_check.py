"""Compare real hardware work inside each synchronized SideIDs microbenchmark marker."""
import argparse
from decimal import Decimal
import gzip
import json
from pathlib import Path


def calc(trace,receipt):
    with (gzip.open if trace.suffix==".gz" else open)(trace,"rt",encoding="utf8") as f:
        data=json.load(f)
    events=data if isinstance(data,list) else data["traceEvents"]
    pids={e["pid"] for e in events if e.get("name")=="process_name" and e.get("args",{}).get("name")=="Ascend Hardware"}
    device=[];markers={}
    for e in events:
        if e.get("ph")!="X":continue
        start=int(Decimal(str(e["ts"]))*1000);end=start+int(Decimal(str(e["dur"]))*1000)
        n=e["name"];args=e.get("args",{})
        if n.startswith("SideIDs/"):
            if n in markers:raise ValueError("duplicate marker "+n)
            markers[n]=(start,end)
        if e["pid"] in pids and str(args.get("Task Type","")).startswith(("AI_","MIX_","SDMA","HDMA","KERNEL_","MEMSET","MEMCPY")):
            device.append((start,end,n,str(args["Task Type"]),args.get("Physic Stream Id"),args.get("Task Id")))
    rec={v["tag"]:v["counts"] for v in json.loads(receipt.read_bytes())["profile"]}
    if set(rec)!=set(markers) or len(markers)!=36:raise ValueError("incomplete 4 x 3 x 3 marker set")
    rows={}
    for name,(lo,hi) in sorted(markers.items()):
        tasks=sorted(t for t in device if lo<=t[0]<=t[1]<=hi)
        cross=[t for t in device if t[1]>lo and t[0]<hi and t not in tasks]
        busy, end = 0, None
        for a,b,*_ in tasks:
            busy += max(0,b-max(a,end if end is not None else a))
            end = max(end if end is not None else b,b)
        span = end-tasks[0][0] if tasks else 0
        rows[name]={"counts":rec[name],"marker_us":(hi-lo)/1000,"crossing":len(cross),
                    "first_to_last_us":span/1000,"between_device_tasks_idle_us":(span-busy)/1000,
                    "device_work_union_us":busy/1000,
                    "device":[{"name":n,"type":k,"stream":s,"id":i,"duration_us":(v-u)/1000} for u,v,n,k,s,i in tasks]}
    checks=[]
    for name,row in rows.items():
        if not name.endswith("/S"):continue
        a=rows[name[:-1]+"A"];p=rows[name[:-1]+"P"]
        key=lambda r:[(t["name"],t["type"],t["stream"]) for t in r["device"]]
        checks.append({"shape_repeat":name.rsplit("/",1)[0],
            "A_count":len(a["device"]),"S_count":len(row["device"]),"P_count":len(p["device"]),
            "S_P_same_work":len(row["device"])==1 and key(row)==key(p) and all(t["stream"] is not None for t in row["device"]),
            "P_native":p["counts"].get("cpp_returned")==1,
            "crossing":a["crossing"]+row["crossing"]+p["crossing"]})
    return {"checks":checks,"windows":rows,"all_verified":all(c["A_count"]>1 and c["S_P_same_work"] and c["P_native"] and c["crossing"]==0 for c in checks)}


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ("trace","receipt","out"):ap.add_argument(n,type=Path)
    a=ap.parse_args();a.out.write_text(json.dumps(calc(a.trace,a.receipt),indent=2)+"\n",encoding="utf8")
