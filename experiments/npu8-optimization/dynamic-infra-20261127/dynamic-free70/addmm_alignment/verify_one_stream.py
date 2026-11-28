"""Rejoin actual Addmm producer, consumer, launch and hardware work inside four one-stream markers."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import tarfile

from packed_profile import Lanes,Span,stats
from analyze_dynamic_trace import is_work


def analyze(archive):
    with tarfile.open(archive) as a:
        files={m.name:a.extractfile(m).read() for m in a if m.isfile()}
    trace,=[v for k,v in files.items() if k.endswith('/trace_view.json')]
    check,=[json.loads(v) for k,v in files.items() if k.endswith('/micro.json')]
    assert check=={"4":True,"5":True}
    obj=json.loads(trace)
    events=obj if isinstance(obj,list) else obj["traceEvents"]
    hw,={e["pid"] for e in events if e.get("name")=="process_name" and e["args"]["name"]=="Ascend Hardware"}
    spans=[(Span.read(e),e) for e in events if e.get("ph")=="X"]
    markers=sorted([s for s,e in spans if s.name.startswith("Projection_")],key=lambda s:s.start)
    work=[s for s,e in spans if e["pid"]==hw and is_work(s.args.get("Task Type",""))]
    owners=Lanes([s for s,e in spans if s.name=="aten::addmm"])
    enq=[s for s,e in spans if s.name=="Enqueue@aclnnAddmm"]
    deq={s.args["correlation_id"]:s for s,e in spans if s.name=="Dequeue@aclnnAddmm"}
    workspace=[s for s,e in spans if s.name=="AscendCL@aclnnAddmmGetWorkspaceSize"]
    acl=Lanes([s for s,e in spans if s.name.startswith("AscendCL@aclrtLaunchKernel")])
    result=[]
    for mark in markers:
        inside=lambda s:mark.start<=s.start<s.end<=mark.end
        tasks=sorted(filter(inside,work),key=lambda s:s.start)
        assert not [s for s in work if s.end>mark.start and s.start<mark.end and not inside(s)]
        qs=sorted(filter(inside,enq),key=lambda s:s.start)
        assert len(qs)==8
        counts=Counter((s.name,s.args["Physic Stream Id"]) for s in tasks)
        assert counts=={("aclnnAddmm_CastAiCore_Cast",47):8,("aclnnAddmm_MatMulCommon_MatMulV2",47):8}
        calls=[]
        for q in qs:
            cid=q.args["correlation_id"]
            parent=owners.parent(q)
            consumer=deq[cid]
            launches=acl.children(consumer)
            assert inside(consumer) and len(launches)==2
            sizes=[s for s in workspace if parent.start<=s.start<s.end<=q.start]
            calls.append(dict(queue_id=cid,prepare_ns=q.start-parent.start,workspace_queries=len(sizes),
                              workspace_ns=sum(s.end-s.start for s in sizes),wait_ns=consumer.start-q.end,
                              service_ns=consumer.end-consumer.start,launch_ns=[s.end-s.start for s in launches]))
        result.append({"marker":mark.name,"workspace_queries":sum(c["workspace_queries"] for c in calls),
            "latency":{k:stats([x for c in calls for x in (c[k] if k=="launch_ns" else [c[k]])])
                       for k in ("prepare_ns","workspace_ns","wait_ns","service_ns","launch_ns")},
            "calls":calls,"device":[(s.name,s.args["Task Type"],s.args["Physic Stream Id"],s.args["Task Id"]) for s in tasks]})
    assert [r["marker"] for r in result]==["Projection_A_4","Projection_B_4","Projection_B_5","Projection_A_5"]
    return {"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),
            "source_hashes":{k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())},"markers":result}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("archive",type=Path)
    cli.add_argument("output",type=Path)
    a=cli.parse_args()
    a.output.write_text(json.dumps(analyze(a.archive),indent=2)+"\n",encoding="utf8")
