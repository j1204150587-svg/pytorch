"""Tally relevant entries in each ProfilerStep of an existing HSTU Level1 recording."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("trace", type=Path)
    cli.add_argument("schedule", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    binary = args.trace.read_bytes()
    trace = json.loads(gzip.decompress(binary) if binary[:2] == b"\x1f\x8b" else binary)
    events = [e for e in (trace if isinstance(trace, list) else trace["traceEvents"]) if e.get("ph")=="X"]
    steps = sorted((e for e in events if re.fullmatch(r"ProfilerStep#\d+", e["name"])), key=lambda e:float(e["ts"]))
    schedule = json.loads(args.schedule.read_text())
    rows = []
    markers = re.compile("torchinductor|torch.compiled.region|compiledfunction|triton|plannedfastlaunch|fast_launch_with_plan|ordered_64", re.I)
    for step in steps:
        number = int(step["name"].split("#")[1])
        begin = float(step["ts"])
        in_step = [e for e in events if begin <= float(e["ts"]) < begin + float(step["dur"])]
        names = Counter(e["name"] for e in in_step)
        queue = Counter(e["name"] for e in in_step if e.get("cat") == "enqueue")
        launch_by_tid = defaultdict(list)
        for e in in_step:
            if e["name"] == "AscendCL@aclrtLaunchKernelWithHostArgs":
                launch_by_tid[e["tid"]].append(float(e["ts"]))
        for timestamps in launch_by_tid.values():
            timestamps.sort()
        op_launches = {}
        for op in ("aclnnEmbedding", "aclnnEmbeddingDenseBackward"):
            enclosing = [e for e in in_step if e["name"] == "Dequeue@" + op]
            total = 0
            for e in enclosing:
                timestamps = launch_by_tid[e["tid"]]
                start = float(e["ts"])
                total += bisect_left(timestamps, start + float(e["dur"])) - bisect_left(timestamps, start)
            op_launches[op] = {"dequeues": len(enclosing), "nested_acl_launches": total}
        rows.append({"step": number, "actual_shape": schedule[number], "all_enqueues":sum(queue.values()),
                     "aclnn_enqueues":sum(v for k,v in queue.items() if k.startswith("Enqueue@aclnn")),
                     "acl_launch_calls":sum(map(len,launch_by_tid.values())), "embedding":op_launches,
                     "matching_trace_names": {s:n for s,n in names.items() if markers.search(s)}})
    args.output.write_text(json.dumps({"trace":str(args.trace), "sha256":hashlib.sha256(binary).hexdigest(),
                                       "steps":rows}, indent=2)+"\n")


if __name__ == "__main__":
    main()
