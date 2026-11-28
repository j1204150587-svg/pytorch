"""Bound the native side-ID preparation from real eager HSTU traces."""
import argparse
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path

OPS = ["aclnnGeScalar", "aclnnLtScalar", "aclnnBitwiseAndTensor", "aclnnAdd", "aclnnInplaceFillScalar", "aclnnSWhere"]


def load_csv(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        return list(csv.DictReader(f))


def ns(x):
    return int(Decimal(str(x)) * 1000)


def collect(trace, attribution):
    with gzip.open(trace, "rt", encoding="utf8") as f:
        obj = json.load(f)
    E = obj if isinstance(obj, list) else obj["traceEvents"]
    events = []
    for e in E:
        if e.get("ph") == "X":
            lo = ns(e["ts"])
            events.append((lo, lo+ns(e["dur"]), e))
    scopes = sorted((a,b,e) for a,b,e in events if e["name"]=="npu_hstu_experiment::slab_side_sum")
    q = load_csv(attribution/"queue_calls.csv.gz")
    works = load_csv(attribution/"work_tasks.csv.gz")
    all_gaps = load_csv(attribution/"device_gaps.csv.gz")
    hw = {(str(e["args"]["Physic Stream Id"]),str(e["args"]["Task Id"])):(a,b,e["name"]) for a,b,e in events
          if e.get("args",{}).get("Task Type") and "Physic Stream Id" in e["args"]}
    free = [(a,b) for a,b,e in events if e["name"]=="Free"]
    def free_ms(a,b):
        return sum(max(0,min(b,end)-max(a,start)) for start,end in free)/1e6
    result=[]
    for slo,shi,scope in scopes:
        step, = [int(e["name"].split("#")[1]) for a,b,e in events if e["name"].startswith("ProfilerStep#") and a<=slo<shi<=b]
        qr = [r for r in q if int(r["step"])==step]
        idx, = [i for i in range(len(qr)-len(OPS)) if [r["op"] for r in qr[i:i+len(OPS)]]==OPS]
        requests = qr[idx:idx+len(OPS)]
        assert qr[idx+len(OPS)]["op"]=="aclnnEmbedding"
        ids={r["queue_id"] for r in requests}
        tasks=[r for r in works if int(r["step"])==step and r["queue_id"] in ids]
        assert len(tasks)==9 and {r["stream"] for r in tasks}=={"47"}
        physical=[hw[(r["stream"],r["task_id"])] for r in tasks]
        assert [r["task"] for r in tasks] == [name for _,_,name in physical]
        moved = min(a for a,b,e in events if slo<=a<b<=shi and e["tid"]==scope["tid"] and e["name"]=="aten::movedim")
        embedding, = [a for a,b,e in events if slo<=a<b<=shi and e["tid"]==scope["tid"] and e["name"]=="aten::embedding"]
        physical_ids = {(t["stream"], t["task_id"]) for t in tasks}
        before = [g for g in all_gaps if int(g["step"])==step and (g["stream"],g["task_id"]) in physical_ids]
        first_id=tasks[0]["task_id"]
        lead=sum(int(g["gap_ns"]) for g in before if g["task_id"]==first_id)
        interior=sum(int(g["gap_ns"]) for g in before if g["task_id"]!=first_id)
        result.append(dict(step=step,shape=[int(qr[0]["batch"]),int(qr[0]["seq_len"])],
            host_movedim_to_embedding_us=(embedding-moved)/1000,host_window_official_free_ms=free_ms(moved,embedding),
            before_first_task_all_stream_gap_ms=lead/1e6, inter_task_all_stream_gaps_ms=interior/1e6,
            first_kernel_start_to_last_kernel_end_ms=(physical[-1][1]-physical[0][0])/1e6,
            kernel_work_ms=sum(b-a for a,b,_ in physical)/1e6,
            tasks=[dict(name=r["task"],stream=r["stream"],slot=r["stream_work_slot"],task_id=r["task_id"]) for r in tasks],
            requests=[{k:r[k] for k in ("op","queue_id","aten_to_enqueue_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","acl_api_sum_ns")} for r in requests]))
    if len(result)!=2:raise RuntimeError("expected two source scopes")
    return {"sha256":hashlib.sha256(trace.read_bytes()).hexdigest(),"steps":result}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for n in ("trace","attribution","output"): cli.add_argument(n,type=Path)
    a=cli.parse_args()
    a.output.write_text(json.dumps(collect(a.trace,a.attribution),indent=2)+"\n",encoding="utf8")
