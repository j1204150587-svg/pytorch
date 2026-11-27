"""Locate the eager ordered fold, every physical stream's preceding work, and official Free."""
import argparse
import gzip
from decimal import Decimal
import hashlib
import json
from pathlib import Path


def ns(value):
    return int(Decimal(str(value)) * 1000)


def interval(event):
    start = ns(event["ts"])
    return start, start + ns(event["dur"])


def overlap(a, b):
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def audit(path, schedule):
    binary = path.read_bytes()
    d = json.loads(gzip.decompress(binary) if binary[:2]==b"\x1f\x8b" else binary)
    events = d if isinstance(d, list) else d["traceEvents"]
    pids = {e["args"]["name"]:e["pid"] for e in events if e.get("name")=="process_name"}
    xs = [e for e in events if e.get("ph")=="X"]
    active_tasks = [e for e in xs if e["pid"]==pids["Ascend Hardware"] and
                    str(e.get("args",{}).get("Task Type","")).startswith(
                        ("KERNEL_", "AI_", "MIX_", "AIC", "AIV", "SDMA", "HDMA", "MEMCPY", "MEMSET"))]
    free = [interval(e) for e in xs if e["pid"]==pids["Overlap Analysis"] and e["name"]=="Free"]
    deqs = {e["args"]["correlation_id"]:e for e in xs if e["name"]=="Dequeue@ordered_64"}
    rows = []
    for step in sorted((e for e in xs if e["name"].startswith("ProfilerStep#")), key=interval):
        s = interval(step)
        belongs = lambda e:s[0] <= interval(e)[0] < s[1]
        host, = [e for e in xs if belongs(e) and e["name"]=="ordered_64" and e.get("cat")=="cpu_op"]
        enqueue, = [e for e in xs if belongs(e) and e["name"]=="Enqueue@ordered_64"]
        kernel, = [e for e in active_tasks if belongs(e) and e["name"]=="ordered_64"]
        dequeue = deqs[enqueue["args"]["correlation_id"]]
        h, q, c, k = map(interval, (host, enqueue, dequeue, kernel))
        previous_end = max(min(interval(e)[1],k[0]) for e in active_tasks if interval(e)[0]<k[0])
        gap = (previous_end, k[0])
        scopes = {"producer_recorded_prefix": (h[0], q[0]), "enqueue":q, "wait":(q[1],c[0]), "consumer":c}
        launches = [{"name":e["name"],"duration_us":e["dur"],"intersect_gap_us":overlap(interval(e),gap)/1e3}
                    for e in xs if e["pid"]==pids.get("CANN") and e["tid"]==dequeue["tid"]
                    and c[0]<=interval(e)[0] and interval(e)[1]<=c[1] and "launch" in e["name"].lower()]
        n = int(step["name"].split("#")[-1])
        rows.append(dict(step=n, shape=schedule[n], stream=kernel["args"]["Physic Stream Id"],
                         kind=kernel["args"]["Task Type"], kernel_us=(k[1]-k[0])/1e3,
                         preceding_no_work_us=(gap[1]-gap[0])/1e3,
                         preceding_no_work_official_free_us=sum(overlap(gap,f) for f in free)/1e3,
                         intervals_us={name:{"duration":(i[1]-i[0])/1e3,"within_preceding_gap":overlap(i,gap)/1e3}
                                       for name,i in scopes.items()}, nested_cann_launches=launches,
                         all_intervals_ns={"host":h,"enqueue":q,"dequeue":c,"kernel":k,"previous_gap":gap}))
    return dict(trace=str(path), sha256=hashlib.sha256(binary).hexdigest(),step_rows=rows)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace",type=Path)
    cli.add_argument("schedule",type=Path)
    cli.add_argument("output",type=Path)
    a=cli.parse_args()
    a.output.write_text(json.dumps(audit(a.trace,json.loads(a.schedule.read_text())),indent=2)+"\n")
