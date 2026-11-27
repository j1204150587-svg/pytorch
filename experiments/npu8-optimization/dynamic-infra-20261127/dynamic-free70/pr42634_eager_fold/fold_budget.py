"""Bound the host-side opportunity for the exact eager fold on a real Level0 trace."""
import argparse
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "causal_tril"))
from core_budget import intersect, merge, ns


def analyze(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf-8-sig") as f:
        doc = json.load(f)
    rows = doc["traceEvents"] if isinstance(doc, dict) else doc
    pid, = {r["pid"] for r in rows if r["name"]=="process_name" and r["args"]["name"]=="Ascend Hardware"}
    events = [r for r in rows if r["ph"] == "X"]
    def span(r):
        return ns(r["ts"]), ns(r["ts"]) + ns(r["dur"])
    def named(name):
        return sorted([r for r in events if r["name"] == name], key=lambda x: ns(x["ts"]))
    free = merge([span(r) for r in named("Free")])
    hw = sorted([r for r in events if r["pid"] == pid], key=lambda r: span(r)[0])
    steps = [r for r in events if r["name"].startswith("ProfilerStep#")]
    output = []
    for s in steps:
        a,b = span(s)
        def choose(rs):
            v, = [r for r in rs if a <= span(r)[0] < b]
            return v
        owner = choose(named("OrderedTableSum"))
        enq = choose(named("Enqueue@ordered_64"))
        deq = choose(named("Dequeue@ordered_64"))
        task = choose([r for r in hw if r["name"]=="ordered_64"])
        if enq["args"]["correlation_id"] != deq["args"]["correlation_id"]:
            raise ValueError("unmatched queue")
        h, q, d, k = map(span,(owner,enq,deq,task))
        last = max([r for r in hw if span(r)[0] < k[0]], key=lambda r: span(r)[1])
        # If another task spans the fold start, the global device gap is empty.
        gap_start = min(k[0],span(last)[1])
        before = [(h[0],q[0])]
        aten = merge([span(r) for r in events if r.get("cat")=="cpu_op" and r["tid"]==owner["tid"]
                     and r["name"].startswith("aten::") and h[0] <= span(r)[0] and span(r)[1] <= q[0]])
        output.append({"step": s["name"], "kernel": task["name"], "stream": task["args"]["Physic Stream Id"],
            "device_submissions":1, "previous_hw":last["name"], "kernel_us":(k[1]-k[0])/1e3,
            "owner_us":(h[1]-h[0])/1e3, "owner_to_enqueue_us":(q[0]-h[0])/1e3,
            "before_enqueue_child_ATen_us":sum(y-x for x,y in aten)/1e3,
            "owner_to_enqueue_official_free_us":intersect(before,free)/1e3,
            "pre_kernel_global_hw_gap_us":(k[0]-gap_start)/1e3,
            "owner_to_enqueue_intersect_pre_kernel_gap_us":intersect(before,[(gap_start,k[0])])/1e3,
            "enqueue_us":(q[1]-q[0])/1e3, "queue_wait_us":(d[0]-q[1])/1e3,
            "dequeue_us":(d[1]-d[0])/1e3})
    return output


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("trace",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    report=analyze(args.trace)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report))
