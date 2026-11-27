"""Match raw ACL launches inside paired user markers to device tasks using CANN's HostToDevice flow."""
import argparse
from collections import defaultdict
from decimal import Decimal
import gzip
import json
from pathlib import Path


def ns(x):
    return int(Decimal(str(x))*1000)


def extract(path):
    with (gzip.open if path.suffix==".gz" else open)(path,"rt",encoding="utf8") as f:
        data=json.load(f)
    events=data["traceEvents"] if isinstance(data,dict) else data
    pids={e["args"]["name"]:e["pid"] for e in events if e["name"]=="process_name"}
    src=defaultdict(list); dst=defaultdict(list); markers=[]; apis=[]; hardware=[]; query=[]
    for e in events:
        if e.get("cat")=="HostToDevice":
            if e["ph"]=="s" and e["pid"]==pids["CANN"]:
                src[e["id"]].append((e["tid"],ns(e["ts"])))
            if e["ph"]=="f" and e["pid"]==pids["Ascend Hardware"]:
                dst[e["tid"],ns(e["ts"])].append(e["id"])
        if e["ph"]!="X":
            continue
        tid=e["tid"]; t=ns(e["ts"]); name=e["name"]; end=t+ns(e["dur"])
        if name.startswith("RepeatProbe/"):
            markers.append((t,end,tid,name))
        if e["pid"]==pids["CANN"] and name.startswith(("AscendCL@aclrtLaunchKernel","AscendCL@aclrtMemcpy")):
            apis.append((t,end,tid,name))
        if e["pid"]==pids["CANN"] and name.endswith("aclnnEmbeddingDenseBackwardGetWorkspaceSize"):
            query.append(t)
        if e["pid"]==pids["Ascend Hardware"] and "Task Type" in e.get("args",{}):
            tt=e["args"]["Task Type"]
            if tt.startswith(("KERNEL_","AI_","MIX_","SDMA","HDMA","MEMCPY","MEMSET")):
                hardware.append((t,tid,e["args"]["Physic Stream Id"],tt,name))
    rows=[]
    for start,end,tid,name in sorted(markers):
        calls=[op for left,right,th,op in sorted(apis) if th==tid and start<=left<right<=end]
        tasks=[]
        for t,ht,stream,kind,task in sorted(hardware):
            ids=dst[ht,t]
            if len(ids)==1 and len(src[ids[0]])==1:
                st,stamp=src[ids[0]][0]
                if st==tid and start<=stamp<=end:
                    tasks.append(dict(stream=stream,type=kind,name=task))
        rows.append(dict(marker=name,acl_apis=calls,tasks=tasks,
                         fully_matched=bool(calls) and len(calls)==len(tasks)))
    return rows,len(query)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a",type=Path); parser.add_argument("b",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--compare-geometries",action="store_true",
                        help="match request ordinal across raw-family (stock) and bounded (repeat)")
    args=parser.parse_args()
    a,aq=extract(args.a); b,bq=extract(args.b)
    def identity(name):
        return name.split("/")[-1] if args.compare_geometries else name.replace("/A/","/").replace("/B/","/")
    checks=[dict(pair=i,a=x,b=y,matched=x["fully_matched"] and y["fully_matched"] and
                identity(x["marker"])==identity(y["marker"]) and
                x["acl_apis"]==y["acl_apis"] and x["tasks"]==y["tasks"]) for i,(x,y) in enumerate(zip(a,b))]
    full=len(a)==len(b)==8 and all(row["matched"] for row in checks)
    args.out.write_text(json.dumps(dict(a_queries=aq,b_queries=bq,a_markers=len(a),b_markers=len(b),
        same_real_submissions_and_stream=full,comparisons=checks),indent=2))
    return 0 if full else 1


if __name__=="__main__":
    raise SystemExit(main())
