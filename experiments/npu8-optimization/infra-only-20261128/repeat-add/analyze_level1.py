"""Join each marked Add queue item to its consumer, ACL launch and synchronized hardware interval."""
import argparse
from collections import Counter
from decimal import Decimal
import gzip
import json
import math
from pathlib import Path
import statistics


def analyze(path):
    op=gzip.open if path.suffix==".gz" else open
    with op(path,"rt",encoding="utf-8") as f:
        data=json.load(f)
    events=data if isinstance(data,list) else data["traceEvents"]
    hardware,={e["pid"] for e in events if e["name"]=="process_name" and e["args"]["name"]=="Ascend Hardware"}
    x=[]
    for e in events:
        if e.get("ph")=="X":
            l=int(Decimal(str(e["ts"]))*1000)
            x.append(dict(e,l=l,r=l+int(Decimal(str(e["dur"]))*1000)))
    def inside(v,p):
        return p["l"]<=v["l"]<=v["r"]<=p["r"]
    def d(v):
        return v["r"]-v["l"]
    def pct(values):
        v=sorted(values)
        return dict(median=statistics.median(v),p95=v[math.ceil(.95*len(v))-1],max=v[-1])
    out=[]
    for mark in sorted((e for e in x if e["name"].startswith("RepeatAdd_")),key=lambda k:k["l"]):
        local=[e for e in x if inside(e,mark)]
        queue=[e for e in local if e["name"]=="Enqueue@aclnnAdd"]
        dq={e["args"]["correlation_id"]:e for e in local if e["name"]=="Dequeue@aclnnAdd"}
        device=[e for e in local if e["pid"]==hardware and e["name"].startswith("aclnnAdd_")]
        streams={e["args"]["Physic Stream Id"] for e in device}
        if not len(queue)==len(dq)==len(device)==8 or len(streams)!=1:
            raise ValueError((mark["name"],len(queue),len(dq),len(device),streams))
        rows=[]
        for q in queue:
            a,=[e for e in local if e["name"]=="aten::add" and e["tid"]==q["tid"] and inside(q,e)]
            de=dq[q["args"]["correlation_id"]]
            acl,=[e for e in local if e["name"].startswith("AscendCL@aclrtLaunchKernel") and e["tid"]==de["tid"] and inside(e,de)]
            rows.append(dict(producer_pre_enqueue=(q["l"]-a["l"])/1000,enqueue=d(q)/1000,
                             queue_wait=(de["l"]-q["r"])/1000,consumer=d(de)/1000,
                             consumer_pre_acl=(acl["l"]-de["l"])/1000,acl_launch=d(acl)/1000,
                             consumer_post_acl=(de["r"]-acl["r"])/1000))
        out.append(dict(marker=mark["name"],stream=streams.pop(),launches=len(rows),operator=len(queue),
                        work=Counter(e["name"] for e in device),rows_us=rows,
                        workspace_queries=sum(e["name"]=="AscendCL@aclnnAddGetWorkspaceSize" for e in local),
                        statistics_us={k:pct([v[k] for v in rows]) for k in rows[0]}))
    if len(out)!=4 or len({v["stream"] for v in out})!=1 or any(v["work"]!={"aclnnAdd_AddAiCore_Add":8} for v in out):
        raise ValueError(out)
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    a.output.write_text(json.dumps(analyze(a.trace),indent=2)+"\n")
