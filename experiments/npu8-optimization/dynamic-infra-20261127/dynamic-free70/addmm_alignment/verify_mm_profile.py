"""Recompute cold and revisited MM preparation, queue, consumer, ACL and real hardware work."""
import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tarfile

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"analysis"))
from analyze_dynamic_trace import Lanes, Span, stats, is_work


def analyze(archive,schedule,chip_log):
    with tarfile.open(archive) as packed:
        files={m.name:packed.extractfile(m).read() for m in packed if m.isfile()}
    def get(prefix, tail):
        x,=[raw for name,raw in files.items() if name.startswith(prefix) and name.endswith(tail)]
        return x
    raw=get("mm-rebase-","/trace_view.json")
    obj=json.loads(raw)
    events=obj if isinstance(obj,list) else obj["traceEvents"]
    hardware,={e["pid"] for e in events if e["name"]=="process_name" and e["args"]["name"]=="Ascend Hardware"}
    spans=[(Span.read(e),e) for e in events if e.get("ph")=="X"]
    atens=Lanes([s for s,e in spans if s.name=="aten::mm"])
    api=Lanes([s for s,e in spans if s.name.startswith("AscendCL@aclrtLaunchKernel")])
    work=[s for s,e in spans if e["pid"]==hardware and is_work(s.args.get("Task Type",""))]
    issues=[s for s,e in spans if s.name=="Enqueue@aclnnMm"]
    drains={s.args["correlation_id"]:s for s,e in spans if s.name=="Dequeue@aclnnMm"}
    size=[s for s,e in spans if s.name=="AscendCL@aclnnMmGetWorkspaceSize"]
    markers=sorted([s for s,e in spans if s.name.startswith("ReadOnlyMM_")],key=lambda x:x.start)
    seen=Counter();rows=[]
    for marker in markers:
        _,leg,step=marker.name.split("_")
        seen[leg,step]+=1
        inside=lambda x: marker.start<=x.start<x.end<=marker.end
        tasks=sorted(filter(inside,work),key=lambda x:x.start)
        cross=[s for s in work if s.start<marker.end and s.end>marker.start and not inside(s)]
        calls=[]
        for q in sorted(filter(inside,issues),key=lambda x:x.start):
            cid=q.args["correlation_id"]
            owner=atens.parent(q)
            consumer=drains[cid]
            launches=api.children(consumer)
            assert owner is not None and inside(consumer) and len(launches)==1
            queries=[w for w in size if owner.start<=w.start<w.end<=q.start]
            calls.append(dict(queue=cid,prepare_ns=q.start-owner.start,workspace_count=len(queries),
                              workspace_ns=sum(x.end-x.start for x in queries),wait_ns=consumer.start-q.end,
                              service_ns=consumer.end-consumer.start,launch_ns=launches[0].end-launches[0].start))
        assert len(calls)==len(tasks)==8 and not cross
        assert sum(x["workspace_count"] for x in calls)==sum(inside(w) for w in size)
        rows.append(dict(variant=leg,train_step=int(step),shape=schedule[int(step)],visit=seen[leg,step],
                         marker_us=(marker.end-marker.start)/1000,workspace_queries=sum(inside(w) for w in size),
                         calls=calls,latency={k:stats([x[k+"_ns"] for x in calls])
                                            for k in ("prepare","workspace","wait","service","launch")},
                         device=[{"task":s.name,"type":s.args["Task Type"],"physical_stream":s.args["Physic Stream Id"],
                                  "task_id":s.args["Task Id"],"us":(s.end-s.start)/1000} for s in tasks]))
    assert [r["variant"] for r in rows]==list("AABBBBAA")
    assert [r["train_step"] for r in rows]==[4,5]*4
    signature={(t["task"],t["type"],t["physical_stream"]) for r in rows for t in r["device"]}
    assert len(signature)==1
    micro=json.loads(get("mm-rebase-","/micro.json"))
    assert micro["exact"] and [r["shape"] for r in micro["shape_checks"]]==schedule
    assert all(r["differing_bf16_words"]==[0]*8 for r in micro["shape_checks"])
    for i,row in enumerate(micro["windows"]):
        assert row["us_per_eight_mm"]==json.loads(get("mm-rebase-",f"/mm-{i}-{row['leg']}.json"))["us_per_eight_mm"]
    addmm=json.loads(get("native-addmm-","/micro.json"))
    a,b=[json.loads(get("native-addmm-","/"+name)) for name in ("gate-0-A.json","gate-1-B.json")]
    assert a["library_sha256"]==b["library_sha256"] and a["invalid_hash"]==b["invalid_hash"]
    assert a["fallback"]==0 and b["fallback"]==1 and a["words"]==b["words"]
    assert len(a["words"])==480 and all(len(x)==4 for x in a["words"])
    assert a["counters"]==[0]*4 and b["counters"]==[480,0,960,220]
    assert addmm["gate_exact"] and not addmm["differing_projections"]
    for i,row in enumerate(addmm["timing"]):
        measurement=json.loads(get("native-addmm-",f"/timing-{i}-{row['route']}.json"))
        assert row["us_per_eight_addmm"]==measurement["us_per_eight_addmm"] and row["counters"]==measurement["counters"]
    samples=[json.loads(line) for line in chip_log.read_bytes().splitlines() if line]
    occupancy={}
    for name,expected in (("native-addmm-",9),("mm-rebase-",8)):
        a,b=[json.loads(get(name,"/"+boundary+".card.json")) for boundary in ("before","after")]
        assert all(c["health"]=="OK" and c["logical_device"]==c["physical_id"]==8 and c["board"]==4
                   and c["chip"]==0 and not c["host_pids"] and c["orphan_context"] is None for c in (a,b))
        lo,hi=[datetime.fromisoformat(c["utc"]).timestamp() for c in (a,b)]
        relevant=[r for r in samples if lo<=datetime.fromisoformat(r["utc"]).timestamp()<hi+1]
        assert len(relevant)==expected and all(r["valid"] and all(p["owner"]=="training" for p in r["processes"]) for r in relevant)
        occupancy[name]={"before":a["utc"],"after":b["utc"],"sample_count":len(relevant)}
    return {"source_sha256":{k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items())},
            "trace_sha256":hashlib.sha256(raw).hexdigest(),"exact_projections":480,
            "occupancy_sha256":hashlib.sha256(chip_log.read_bytes()).hexdigest(),"chip_samples":occupancy,
            "windows":micro["windows"],"physical_signature":list(signature),"visits":rows,
            "native_addmm":addmm,"native_library_sha256":json.loads(get("native-addmm-","/gate-0-A.json"))["library_sha256"]}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("archive","schedule","chip_log","output"):
        cli.add_argument(n,type=Path)
    args=cli.parse_args()
    result=analyze(args.archive,json.loads(args.schedule.read_bytes()),args.chip_log)
    args.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
