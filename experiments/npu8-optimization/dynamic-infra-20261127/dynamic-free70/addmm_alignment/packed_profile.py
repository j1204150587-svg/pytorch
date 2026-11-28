"""Use measured producer, consumer and per-stream task anchors in the two cold Addmm profiles."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"analysis"))
from analyze_dynamic_trace import Lanes, Span, stats


def profile(events):
    events=events if isinstance(events,list) else events["traceEvents"]
    hw,=[e["pid"] for e in events if e["name"]=="process_name" and e["args"]["name"]=="Ascend Hardware"]
    spans=[(Span.read(e),e) for e in events if e.get("ph")=="X"]
    atens=Lanes([s for s,e in spans if s.name=="aten::addmm"])
    queues=[s for s,e in spans if s.name=="Enqueue@aclnnAddmm"]
    consumes={s.args["correlation_id"]:s for s,e in spans if s.name=="Dequeue@aclnnAddmm"}
    launch=Lanes([s for s,e in spans if s.name.startswith("AscendCL@aclrtLaunchKernel")])
    output=[]
    for step,shape in ((4,[1,19]),(5,[2,14])):
        mark,=[s for s,e in spans if s.name==f"AddmmProbe_{step}"]
        inside=lambda s: mark.start<=s.start<s.end<=mark.end
        ws=[s for s,e in spans if s.name=="AscendCL@aclnnAddmmGetWorkspaceSize" and inside(s)]
        row=[]
        for producer in sorted(filter(inside,queues),key=lambda s:s.start):
            owner=atens.parent(producer)
            consumer=consumes[producer.args["correlation_id"]]
            calls=launch.children(consumer)
            row.append(dict(slot=len(row),queue=producer.args["correlation_id"],prepare_ns=producer.start-owner.start,
                            workspace_ns=sum(s.end-s.start for s in ws if owner.start<=s.start<s.end<=producer.start),
                            wait_ns=consumer.start-producer.end,service_ns=consumer.end-consumer.start,
                            acl_launch_ns=[s.end-s.start for s in calls]))
        tasks=[(s.args["Physic Stream Id"],s.name) for s,e in sorted(spans,key=lambda p:p[0].start)
               if e["pid"]==hw and s.name.startswith("aclnnAddmm_") and inside(s)]
        if len(row)!=8 or len(tasks)!=16 or any(len(r["acl_launch_ns"])!=2 for r in row):
            raise ValueError("truncated profile or unexpected physical work")
        output.append(dict(step=step,shape=shape,workspace_count=len(ws),calls=row,tasks=tasks,
                           prepare=stats([x["prepare_ns"] for x in row]),wait=stats([x["wait_ns"] for x in row]),
                           service=stats([x["service_ns"] for x in row]),acl=stats([t for x in row for t in x["acl_launch_ns"]])))
    return output


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    data={}
    with tarfile.open(args.archive) as archive:
        for member in archive:
            if member.name.endswith("/trace_view.json"):
                mode=Path(member.name).parts[1].split("-")[2]
                raw=archive.extractfile(member).read()
                data[mode]={"source":member.name,"sha256":hashlib.sha256(raw).hexdigest(),"steps":profile(json.loads(raw))}
    if {s["step"] for s in data["A"]["steps"]}!={4,5}:
        raise ValueError("missing profiled shapes")
    data["tasks_same_by_step"]=[a["tasks"]==b["tasks"] for a,b in zip(data["A"]["steps"],data["B"]["steps"],strict=True)]
    args.output.write_text(json.dumps(data,indent=2))
