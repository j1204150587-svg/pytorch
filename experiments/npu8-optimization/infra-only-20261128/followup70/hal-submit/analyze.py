"""Join rank-local HAL calls to same-thread ACL intervals and unique device flows."""
import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import struct
import sys

cli = argparse.ArgumentParser()
cli.add_argument("leg", type=Path)
cli.add_argument("capture", type=Path)
cli.add_argument("analyzer", type=Path)
cli.add_argument("output", type=Path)
cli.add_argument("--clock",type=Path,help="independent simultaneous wall/raw clock brackets")
args = cli.parse_args()
sys.path.insert(0, str(args.analyzer.parent))
from analyze_dynamic_trace import Span, Lanes, is_work, ns, stats, write_csv

with gzip.open(args.leg / "trace_view.json.gz") as f:
    trace = json.load(f)
events = trace if isinstance(trace, list) else trace["traceEvents"]
pids = {e["args"]["name"]: e["pid"] for e in events if e["name"] == "process_name"}
steps, starts, ends, anchors = {}, defaultdict(list), defaultdict(list), defaultdict(list)
launches, tasks, dequeues = [], [], []
for e in events:
    pid, phase = e["pid"], e["ph"]
    if e.get("cat") == "HostToDevice":
        if phase == "s" and pid == pids["CANN"]:
            starts[e["id"]].append((e["tid"], ns(e["ts"])))
        if phase == "f" and pid == pids["Ascend Hardware"]:
            ends[e["tid"], ns(e["ts"])].append(e["id"])
    if phase != "X":
        continue
    s = Span.read(e)
    if s.name.startswith("ProfilerStep#"):
        steps[int(s.name.split("#")[1])] = s
    if e.get("cat") == "dequeue":
        dequeues.append(s)
    if pid == pids["CANN"]:
        anchors[s.tid, s.start, s.args.get("connection_id")].append(s)
        if s.name.startswith(("AscendCL@aclrtLaunchKernel", "AscendCL@aclrtMemcpy")):
            launches.append(s)
    if pid == pids["Ascend Hardware"] and is_work(s.args.get("Task Type", "")):
        tasks.append(s)

shape = {s["step"]: [s["batch_size"], s["seq_len"]] for s in
         json.loads((args.leg / "rank000/metrics.json").read_bytes())["steps"]}
clock_meta = {}
def trace_time(t):
    return t
if args.clock:
    src, = (args.leg/"rank000/profile").rglob("host/start_info")
    info = json.loads(src.read_bytes())
    epoch = int(info["collectionTimeBegin"])*1000-int(info["clockMonotonicRaw"])
    obs = [json.loads(line) for line in args.clock.read_bytes().splitlines() if line]
    clocks = [(a+c)//2 for a,b,c in obs]
    diffs = [b-(a+c)//2 for a,b,c in obs]
    assert clocks == sorted(clocks)
    def trace_time(t):
        i = bisect_left(clocks,t)
        if i==0 or i==len(clocks):
            return None
        span = clocks[i]-clocks[i-1]
        if span>500_000_000:
            raise ValueError("host clock samples have a gap larger than 500 ms")
        drift = (diffs[i]-diffs[i-1])*(t-clocks[i-1])//span
        return t+diffs[i-1]+drift+epoch
    clock_meta = dict(source=str(src),profiler_offset_ns=epoch,samples=len(obs),max_bracket_ns=max(c-a for a,b,c in obs),
                      first_wall_ns=clocks[0],last_wall_ns=clocks[-1],mapping="interpolate simultaneous CLOCK_REALTIME to MONOTONIC_RAW; add native CANN host/start_info offset")
apis, queues = Lanes(launches), Lanes(dequeues)
physical = defaultdict(list)
frontier = None
by_start = defaultdict(list)
for task in tasks:
    by_start[task.start].append(task)
gaps = {}
for t, group in sorted(by_start.items()):
    if frontier is not None and frontier < t and len(group) == 1:
        gaps[id(group[0])] = (frontier, t)
    frontier = max(frontier or 0, *(s.end for s in group))
for task in tasks:
    flows = ends[task.tid, task.start]
    if len(flows) != 1 or len(starts[flows[0]]) != 1:
        continue
    tid, t = starts[flows[0]][0]
    anchor = anchors[tid, t, task.args.get("connection_id")]
    if len(anchor) != 1:
        continue
    linked = apis.children(anchor[0])
    if len(linked) == 1:
        physical[id(linked[0])].append(task)

rows, files = [], []
record = struct.Struct("<QQIIIIIi")
for file in sorted(args.capture.glob(f"{pids['Python']}-*.bin")):
    tid = int(file.stem.split("-")[1])
    with file.open("rb") as f:
        capacity, used, dropped, size = struct.unpack("<4Q", f.read(32))
        assert size == record.size and used <= capacity and dropped == 0
        entries = list(record.iter_unpack(f.read(used * size)))
        assert len(entries) == used
    files.append(dict(file=file.name, used=used, dropped=dropped))
    for wall_start, wall_end, device, ts, sq, count, position, status in entries:
        assert wall_end >= wall_start
        start, end = trace_time(wall_start), trace_time(wall_end)
        if start is None or end is None:
            continue
        step = next((n for n, s in steps.items() if s.start <= start < end <= s.end), None)
        if step is None:
            continue
        row = dict(step=step, shape=shape[step], tid=tid, start_ns=start, end_ns=end, wall_start_ns=wall_start, wall_end_ns=wall_end,
                   ns=end-start, sq_id=sq, ts_id=ts, device=device, sqes=count, return_code=status,
                   driver_position=position, acl="", op="", stream="", task="", task_id="",
                   gap_ns=None, hal_in_gap_ns=None, after_hal_ns=None, inside_acl=False)
        probe = Span(start, end, tid, "halSqTaskSend", {})
        api = apis.parent(probe)
        queue = queues.parent(probe)
        if queue:
            row["op"] = queue.name
        if api:
            row.update(acl=api.name, inside_acl=True)
            candidates = physical[id(api)]
            if len(candidates) == 1:
                task = candidates[0]
                row.update(stream=task.args["Physic Stream Id"], task=task.name, task_id=task.args.get("Task Id"),
                           device_start_ns=task.start, device_end_ns=task.end)
                hole = gaps.get(id(task))
                if hole and steps[step].start <= hole[0] < hole[1] <= steps[step].end:
                    left, right = hole
                    row.update(gap_ns=right-left, gap_left_ns=left, gap_right_ns=right,
                               hal_in_gap_ns=max(0, min(right,end)-max(left,start)),
                               after_hal_ns=max(0, right-max(left,end)))
            row["acl_start_ns"] = api.start
        rows.append(row)

args.output.mkdir(parents=True, exist_ok=True)
write_csv(args.output / "calls.csv.gz", rows, compressed=True)
out = dict(files=files, clock=clock_meta,profiler_pid=pids["Python"], method="same TID; complete HAL interval inside ACL; unique device flow", steps={})
for step, scope in sorted(steps.items()):
    group = [r for r in rows if r["step"] == step]
    linked = [r for r in group if r["inside_acl"]]
    multiplicity = Counter((r["tid"], r["acl_start_ns"]) for r in linked)
    singles = [r for r in linked if multiplicity[r["tid"],r["acl_start_ns"]] == 1 and r["gap_ns"] is not None]
    def describe(rs):
        return dict(calls=len(rs), sqes=sum(r["sqes"] for r in rs), duration=stats([r["ns"] for r in rs]))
    sources = sorted({(str(r["stream"]),r["op"]) for r in singles})
    out["steps"][step] = dict(shape=shape[step], all=describe(group), same_tid_inside_acl=describe(linked),
        unique_device_link=sum(r["stream"]!="" for r in linked), unique_gap_and_single_submit=describe(singles),
        return_codes=dict(Counter(r["return_code"] for r in group)),
        sqes_per_call=dict(Counter(r["sqes"] for r in group)),
        physical_stream_per_sq=dict(Counter(f"{r['sq_id']} -> {r['stream']}" for r in linked if r["stream"]!="")),
        linked_acl_multiplicity=dict(Counter(multiplicity.values())),
        nominal_hal_in_successor_gap_ms=sum(r["hal_in_gap_ns"] for r in singles)/1e6,
        nominal_after_hal_in_successor_gap_ms=sum(r["after_hal_ns"] for r in singles)/1e6,
        by_stream_and_op={f"{st} {op}": describe([r for r in singles if str(r["stream"])==st and r["op"]==op]) for st,op in sources})
(args.output/"result.json").write_text(json.dumps(out, indent=2)+"\n")
if not files or not rows or any(s["same_tid_inside_acl"]["calls"] == 0 for s in out["steps"].values()):
    raise RuntimeError("HAL-to-profiler mapping lacks a same-thread time anchor in at least one step")
