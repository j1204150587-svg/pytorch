"""Join stock Level1 kernels, requests and exposed gaps by profiler step/stream/task."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path


def table(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def summary(vals):
    vals.sort()
    if not vals:
        return {}
    def q(p):
        rank = (len(vals)-1)*p
        lo = int(rank)
        return round(vals[lo] + (vals[min(lo+1,len(vals)-1)]-vals[lo])*(rank-lo),3)
    return {"n":len(vals), "p50_us":q(.5), "p95_us":q(.95), "p99_us":q(.99), "max_us":q(1)}


def parse(root, case):
    traces = list((root/"evidence"/case).rglob("kernel_details.csv"))
    if len(traces) != 1:
        raise ValueError(traces)
    kernels = {(x["Step Id"],x["Stream ID"],x["Task ID"]):x for x in table(traces[0])}
    def context(name):
        return table(root/"analysis"/case/(name+".csv.gz"))
    tasks = context("work_tasks")
    calls = {(x["step"],x["queue_id"]):x for x in context("queue_calls")}
    gaps = {(x["step"],x["stream"],x["task_id"]):x for x in context("device_gaps")}
    results = []
    for step in sorted({t["step"] for t in tasks},key=int):
        step_tasks = [t for t in tasks if t["step"]==step]
        b,l = map(int,(step_tasks[0]["batch"],step_tasks[0]["seq_len"]))
        n = b*l
        def key(t):
            return (step,t["stream"],t["task_id"])
        def op(t):
            return calls.get((step,t["queue_id"]),{}).get("op")
        groups = {"divs_side":set(), "mm_dinput_projection":set(),"addmm_projection":set()}
        for t in step_tasks:
            k = kernels.get(key(t),{})
            dims = k.get("Input Shapes","").strip('"')
            if op(t)=="aclnnDivs" and k.get("Type")=="BroadcastTo" and dims.split(";")[0]==f"{b},{l},1,64":
                groups["divs_side"].add(t["queue_id"])
            elif op(t)=="aclnnMm" and k.get("Type")=="MatMulV2" and dims in {f"{n},64;64,192",f"{n},256;256,64"}:
                groups["mm_dinput_projection"].add(t["queue_id"])
            elif op(t)=="aclnnAddmm" and k.get("Type")=="MatMulV2":
                groups["addmm_projection"].add(t["queue_id"])
        for name, ids in groups.items():
            qt = [calls[(step,q)] for q in ids]
            wt = [t for t in step_tasks if t["queue_id"] in ids]
            gp = [gaps[key(t)] for t in wt if key(t) in gaps]
            wt_k = [kernels[key(t)] for t in wt if key(t) in kernels]
            metrics = {}
            for col in ("aten_to_enqueue_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","acl_api_sum_ns","workspace_ns"):
                metrics[col] = summary([int(x[col])/1000 for x in qt if x[col]])
            gcols = ("before_aten_ns","aten_before_workspace_ns","workspace_ns","aten_to_enqueue_remainder_ns","enqueue_ns",
                     "visible_queue_wait_ns","consumer_before_api_ns","acl_submission_api_ns","after_api_ns","gap_ns")
            results.append({"case":case,"group":name,"step":int(step),"shape":[b,l],"shape_first_step":step_tasks[0]["first_step_with_input_shape"],
                "requests":len(ids),"stream_tasks":dict(Counter(t["stream"] for t in wt)), "work":dict(Counter(t["task"] for t in wt)),
                "acl_apis":sum(int(x["launch_api_count"]) for x in qt),"workspace_calls":sum(int(x["workspace_ns"] or 0)>0 for x in qt),
                "latency_us":metrics, "gap_intersection_ms":{col:round(sum(int(x[col] or 0) for x in gp)/1e6,6) for col in gcols},
                "kernel_ms":{tp:round(sum(float(x["Duration(us)"]) for x in wt_k if x["Type"]==tp)/1000,6) for tp in sorted({x["Type"] for x in wt_k})},
                "gap_before_kernel_ms":{task:round(sum(int(x["gap_ns"]) for x in gp if x["task"]==task)/1e6,6) for task in sorted({t["task"] for t in wt})}})
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    base = Path(__file__).resolve().parents[2]
    ap.add_argument("--root",type=Path,default=base/"dynamic-infra-20261127")
    ap.add_argument("--output",type=Path,default=Path(__file__).with_name("stock.json"))
    args = ap.parse_args()
    args.output.write_text(json.dumps([x for case in ("baseline-l1","repeat-late") for x in parse(args.root,case)],indent=2)+"\n",encoding="utf-8")
