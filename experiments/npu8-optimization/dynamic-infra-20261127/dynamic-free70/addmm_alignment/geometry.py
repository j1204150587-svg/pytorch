"""Reconstruct single-rank ZeRO flat view offsets and compare Addmm executor geometry keys."""
import argparse
from collections import Counter
import csv
from decimal import Decimal
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import torch


def csv_rows(path):
    with gzip.open(path,"rt",encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))


def workspace_gap_intersection(trace, step, work, gaps, queues):
    with gzip.open(trace,"rt",encoding="utf-8-sig") as f:
        doc=json.load(f)
    rows=doc["traceEvents"] if isinstance(doc,dict) else doc
    ns=lambda v:int(Decimal(str(v))*1000)
    def span(r):
        s=ns(r["ts"])
        return s,s+ns(r["dur"])
    hardware,={r["pid"] for r in rows if r.get("name")=="process_name" and r["args"]["name"]=="Ascend Hardware"}
    ev=[r for r in rows if r.get("ph")=="X"]
    frame,=[span(r) for r in ev if r["name"]==f"ProfilerStep#{step}"]
    ev=[r for r in ev if frame[0]<=span(r)[0]<frame[1]]
    raw={str(r["args"]["correlation_id"]):r for r in ev if r["name"]=="Enqueue@aclnnAddmm"}
    slots={}
    for q in queues:
        e=raw[q["queue_id"]]
        t=span(e)[0]
        owner,=[span(r) for r in ev if r["name"]=="aten::addmm" and r["tid"]==e["tid"] and span(r)[0]<=t<span(r)[1]]
        s,=[span(r) for r in ev if r["name"]=="AscendCL@aclnnAddmmGetWorkspaceSize" and r["tid"]==e["tid"]
                                 and owner[0]<=span(r)[0] and span(r)[1]<=t]
        slots[q["queue_id"]]=s
    key=lambda r:(r["task"],r["stream"],r["task_id"])
    physical={(r["name"],str(r["args"].get("Physic Stream Id")),str(r["args"].get("Task Id"))):span(r)
                for r in ev if r["pid"]==hardware and r["name"].startswith("aclnnAddmm_")}
    qids={key(r):r["queue_id"] for r in work}
    total=0
    for g in gaps:
        k=key(g)
        start=physical[k][0]
        gs=start-int(g["gap_ns"])
        a,b=slots[qids[k]]
        total+=max(0,min(start,b)-max(gs,a))
    return total/1e6


def report(evidence,profile):
    ws=evidence/"workspace"
    params=evidence/"02-B"/"rank000"
    source=ws/"model.py"
    spec=importlib.util.spec_from_file_location("geometry_original_hstu",source)
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    summary=json.loads((params/"model_summary.json").read_bytes())
    side=json.loads((params/"side_slab_counts.json").read_bytes())
    schedule=json.loads((ws/"configs/dynamic_shapes_npu8.json").read_bytes())
    model=m.MultiTableHSTU(m.ModelConfig(**summary["config"])).bfloat16()
    group=list(model.parameters())
    flat=torch.empty(sum(p.numel() for p in group),dtype=torch.bfloat16)
    views=torch._utils._unflatten_dense_tensors(flat,group)
    for p,q in zip(group,views):
        p.data=q.data
    if (len(group)!=summary["trainable_parameter_tensors"] or flat.numel()!=summary["trainable_parameter_elements"]
            or model.side_embeddings[0].weight.storage_offset()!=side["flat_element_offset"]):
        raise ValueError("CPU reconstruction does not match the direct NPU receipt")
    layers=[]
    for i,block in enumerate(model.blocks):
        for name in ("uvqk","output"):
            linear=getattr(block,name)
            def geometry(t):
                element_offset=t.storage_offset()
                if t.untyped_storage().data_ptr()!=flat.untyped_storage().data_ptr():
                    raise ValueError("projection has separate storage")
                return dict(shape=list(t.shape),stride=list(t.stride()),storage_element_offset=element_offset,
                    storage_elements=t.untyped_storage().nbytes()//t.element_size(),
                    residual_bytes_512=element_offset*t.element_size()%512)
            layers.append(dict(layer=i,projection=name,transposed_weight=geometry(linear.weight.t()),
                               bias=geometry(linear.bias)))
    keys={"original":set(),"aligned_512":set()}
    repeats={k:[] for k in keys}
    trace=[]
    calls=csv_rows(profile/"queue_calls.csv.gz")
    gaps=csv_rows(profile/"device_gaps.csv.gz")
    work=csv_rows(profile/"work_tasks.csv.gz")
    for i,(batch,length) in enumerate(schedule):
        n=batch*length
        for p in layers:
            for method,seen in keys.items():
                geom="storage_element_offset" if method=="original" else "residual_bytes_512"
                # N and the projection imply mat1/output shapes and strides here.
                # Ignore all other key differences, giving an upper bound on reuse.
                key=(n,p["projection"],p["transposed_weight"][geom],p["bias"][geom])
                if key in seen:repeats[method].append(i)
                seen.add(key)
        q=[r for r in calls if r["op"]=="aclnnAddmm" and int(r["step"])==i]
        if not q:continue
        qids={r["queue_id"] for r in q}
        g=[r for r in gaps if r["task"].startswith("aclnnAddmm_") and int(r["step"])==i]
        tasks=[r for r in work if r["queue_id"] in qids and int(r["step"])==i]
        trace.append(dict(step=i,shape=[batch,length],flattened_rows=n,addmm_calls=len(q),
            workspace_observed=sum(bool(r["workspace_ns"]) and int(r["workspace_ns"])>0 for r in q),
            workspace_call_sum_ms=sum(int(r["workspace_ns"] or 0) for r in q)/1e6,
            pre_successor_gaps_ms=sum(int(r["gap_ns"]) for r in g)/1e6,
            complete_clock_chain_gaps=sum(r["state"]=="linked" for r in g),
            workspace_overlap_with_own_successor_gap_ms=workspace_gap_intersection(evidence/"02-B"/"trace_view.json.gz",i,tasks,g,q),
            submitted_tasks=dict(Counter(r["task"] for r in tasks)),physical_streams=dict(Counter(r["stream"] for r in tasks)),
            per_queue=[{"queue_id":int(r["queue_id"]),"projection":layers[j]["projection"],"layer":layers[j]["layer"],
                        "workspace_us":int(r["workspace_ns"] or 0)/1e3,
                        "entry_to_enqueue_us":int(r["aten_to_enqueue_ns"])/1e3} for j,r in enumerate(q)]))
    return dict(model_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),flat=summarize(flat,side,summary),
        projection_views=layers,shapes=len(schedule),unique_flattened_rows=len({b*l for b,l in schedule}),
        projected_keys={k:{"calls":len(schedule)*len(layers),"distinct":len(seen),"eligible_repeats":len(repeats[k]),
                       "eligible_repeats_in_50_step_window":sum(i>=10 for i in repeats[k])} for k,seen in keys.items()},
        profiled_steps=trace)


def summarize(flat,side,summary):
    return dict(elements=flat.numel(),parameters=summary["trainable_parameter_tensors"],
                evidence_side_first=side["group_table_start"],evidence_side_element_offset=side["flat_element_offset"],
                evidence_side_count=side["views_created"],evidence_model=summary["config"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence",type=Path)
    p.add_argument("profile",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.write_text(json.dumps(report(a.evidence,a.profile),indent=2)+"\n")
