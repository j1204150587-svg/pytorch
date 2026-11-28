"""Compare unmodified eager host/queue/device timing for a full Level1 A/B or B/A."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

from exposure import measure


def link_or_copy(source, target):
    if not target.exists():
        try:
            target.hardlink_to(source)
        except OSError:
            shutil.copyfile(source, target)


def files(leg, out):
    out.mkdir(parents=True, exist_ok=True)
    trace = leg / "trace_view.json.gz"
    target = out / "trace_view.json.gz"
    if trace.exists():
        link_or_copy(trace,target)
    else:
        paths = list((leg/"rank000/profile").rglob("trace_view.json"))
        if len(paths) != 1:
            raise ValueError(f"{leg}: expected one Level1 trace, found {paths}")
        with gzip.open(target,"wb",compresslevel=1) as dest, paths[0].open("rb") as source:
            shutil.copyfileobj(source,dest)
    rank = out / "rank000"
    rank.mkdir(exist_ok=True)
    metric = leg / "rank000/metrics.json"
    link_or_copy(metric,rank/"metrics.json")
    return target


def quant(items, key):
    vals = sorted(float(s[key])/1000 for s in items if s.get(key) not in ("",None))
    return {"n":len(vals), **({k:round(vals[int((len(vals)-1)*q)],3) for k,q in
        (("p50_us",.5),("p95_us",.95),("p99_us",.99),("max_us",1))} if vals else {})}


def readcsv(root, name):
    with gzip.open(root/(name+".csv.gz"),"rt",newline="",encoding="utf-8-sig") as source:
        return list(csv.DictReader(source))


def summarize(rows, gaps):
    stages=("aten_to_enqueue_ns","workspace_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","acl_api_sum_ns")
    parts=("before_aten_ns","before_enqueue_unresolved_ns","aten_before_workspace_ns","workspace_ns",
           "aten_to_enqueue_remainder_ns","enqueue_ns","visible_queue_wait_ns","consumer_before_api_ns",
           "acl_submission_api_ns","after_api_ns","gap_ns")
    return dict(calls=len(rows),long_wait_gt100us=sum(float(r["signed_queue_wait_ns"])>100_000 for r in rows),
        latency={k:quant(rows,k) for k in stages},
        next_gap_ms={k:round(sum(int(r[k]) for r in gaps if r.get(k))/1e6,6) for k in parts})


def legdata(leg, dest, analyzer):
    trace = files(leg,dest/"input")
    linked=dest/"linked"
    subprocess.run([sys.executable,"-B",str(analyzer),str(trace.parent),str(linked)],check=True)
    raw=json.loads((linked/"submissions.json").read_bytes())
    exposure={int(v["step"].split("#")[-1]):v for v in measure(trace)}
    calls=readcsv(linked,"queue_calls")
    gaps=readcsv(linked,"device_gaps")
    result={}
    for src in raw["steps"]:
        step=src["step"]
        rr=[r for r in calls if int(r["step"])==step]
        gg=[r for r in gaps if int(r["step"])==step]
        ops=defaultdict(list)
        ogaps=defaultdict(list)
        for r in rr: ops[r["op"]].append(r)
        for g in gg:
            if g.get("op"): ogaps[g["op"]].append(g)
        result[step]=dict(shape=[src["batch"],src["seq_len"]],first_seen=src["first_step_with_input_shape"],
            exposure=exposure[step],physical=src["streams"],hardware=src["hardware_types"],
            acl_api=src["acl_submission_apis"],enqueue_counts=src["enqueue_counts"],
            aten_sequence=src["aten_thread_sequences"],aten_counts=src["aten_counts"],
            all=summarize(rr,gg),ops={op:summarize(v,ogaps[op]) for op,v in sorted(ops.items())})
    return {"trace_sha256":raw["trace_sha256"],"steps":result}


def compare(a,b):
    if a["shape"]!=b["shape"] or a["first_seen"]!=b["first_seen"]:
        raise ValueError("mismatched shapes")
    # Keep stream IDs and full control-event sequences separate from physical work.
    def stream_signature(s):
        return {k:(r["work_n"],r["work_sha256"]) for k,r in s["physical"].items()}
    da,db=stream_signature(a),stream_signature(b)
    sig=lambda s:Counter((r["role"],r["n"],r["sha256"]) for r in s["aten_sequence"])
    return dict(shape=a["shape"],first_seen=a["first_seen"],same_physical_streams_and_work_order=da==db,
                same_work_order_allowing_renumber=sorted(da.values())==sorted(db.values()),
                same_control_and_work=a["physical"]==b["physical"],same_aten_counts=a["aten_counts"]==b["aten_counts"],
                same_aten_per_thread_order=sig(a)==sig(b),same_enqueue_counts=a["enqueue_counts"]==b["enqueue_counts"],
                same_acl_counts=a["acl_api"]==b["acl_api"])


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("run",type=Path)
    cli.add_argument("out",type=Path)
    cli.add_argument("--analyzer",required=True,type=Path)
    args=cli.parse_args()
    legs={p.name[-1]:p for p in args.run.glob("0?-?") if p.is_dir()}
    if set(legs)!={"A","B"}:raise ValueError(f"incomplete pair in {args.run}")
    result={k:legdata(v,args.out/k,args.analyzer) for k,v in legs.items()}
    if set(result["A"]["steps"])!=set(result["B"]["steps"]):raise ValueError("incomplete steps")
    result["compare"]={step:compare(row,result["B"]["steps"][step]) for step,row in result["A"]["steps"].items()}
    (args.out/"pair.json").write_text(json.dumps(result,indent=2)+"\n")
