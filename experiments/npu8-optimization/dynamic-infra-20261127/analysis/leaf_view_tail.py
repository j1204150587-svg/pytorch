"""Identify the repeated differentiable view_as block inside one AccumulateGrad per step."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path

from analyze_dynamic_trace import Span, is_work, stats, write_csv
from inspect_tail import covered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg", type=Path)
    parser.add_argument("analysis", type=Path)
    args = parser.parse_args()
    with gzip.open(args.leg / "trace_view.json.gz", "rt", encoding="utf8") as handle:
        events = json.load(handle)
    if isinstance(events, dict):
        events = events["traceEvents"]
    pids = {e["args"]["name"]:e["pid"] for e in events if e["name"] == "process_name"}
    by_thread, accum, device, free, steps = defaultdict(list), [], [], [], []
    for e in events:
        if e["ph"] != "X":
            continue
        s = Span.read(e)
        if e["pid"] == pids["Ascend Hardware"] and is_work(s.args.get("Task Type", "")):
            device.append(s)
        elif e["pid"] == pids["Overlap Analysis"] and s.name == "Free":
            free.append(s)
        elif s.name.startswith("ProfilerStep#"):
            steps.append(s)
        elif s.name == "torch::autograd::AccumulateGrad":
            accum.append(s)
        else:
            by_thread[s.tid].append(s)
    del events
    metrics = json.loads((args.leg / "rank000/metrics.json").read_bytes())["steps"]
    shapes = {m["step"]: (m["batch_size"],m["seq_len"]) for m in metrics}
    rows = []
    for step in steps:
        number = int(step.name.split("#")[1])
        # Sort by elapsed wall clock then choose the single largest participating leaf hook.
        hook = max((s for s in accum if step.start <= s.start < step.end),key=lambda s:s.end-s.start)
        siblings = [s for s in by_thread[hook.tid] if hook.start <= s.start < s.end <= hook.end]
        views = sorted((s for s in siblings if s.name == "aten::view_as" and s.args["Sequence number"] >= 0),
                       key=lambda s:s.start)
        if not views:
            continue
        left, right = views[0].start, views[-1].end
        block = [s for s in siblings if left <= s.start < s.end <= right]
        counted = Counter(s.name for s in block)
        counters = [s.args["Sequence number"] for s in views]
        def overlap(windows):
            return covered((max(left,s.start),min(right,s.end)) for s in windows if left < s.end and s.start < right) / 1e6
        row = dict(step=number, batch=shapes[number][0],seq_len=shapes[number][1],
                   hook_ms=(hook.end-hook.start)/1e6,block_ms=(right-left)/1e6,
                   first_sequence=counters[0],last_sequence=counters[-1],
                   consecutive=counters==list(range(counters[0],counters[0]+len(counters))),
                   view_as_n=counted["aten::view_as"],view_n=counted["aten::view"],
                   all_other_scopes=json.dumps({k:v for k,v in counted.items() if k not in {"aten::view_as","aten::view"}}),
                   active_device_overlap_ms=overlap(device),official_free_overlap_ms=overlap(free))
        for tag, vals in (("view_as",[s.end-s.start for s in views]),
                          ("inner_view",[s.end-s.start for s in block if s.name=="aten::view"]),
                          ("between",[b.start-a.end for a,b in zip(views,views[1:])])):
            for k,v in stats(vals).items():
                if k != "n":
                    row[tag+"_"+k] = v
        rows.append(row)
    write_csv(args.analysis / "leaf_view_tail.csv",rows)


if __name__ == "__main__":
    main()
