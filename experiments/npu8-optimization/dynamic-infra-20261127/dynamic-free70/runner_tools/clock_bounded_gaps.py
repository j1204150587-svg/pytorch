"""Bound Host/device alignment from observed launches and completed whole-device synchronizations."""
import argparse
from bisect import bisect_right
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "analysis"))
from analyze_dynamic_trace import Lanes, Span, enclosing_aten, is_work, ns, stats, write_csv


def join(leg):
    trace = leg / "trace_view.json.gz"
    with gzip.open(trace, "rt", encoding="utf8") as handle:
        items = json.load(handle)
    events = items if isinstance(items, list) else items["traceEvents"]
    pids = {e["args"]["name"]: e["pid"] for e in events if e["name"] == "process_name"}
    cpu, enq, deq, api, work, syncs, steps, ws = [], {}, {}, [], [], [], {}, []
    sources, destinations, anchors = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in events:
        name, pid, cat = e["name"], e["pid"], e.get("cat")
        if cat == "HostToDevice":
            if e["ph"] == "s" and pid == pids["CANN"]:
                sources[e["id"]].append((e["tid"], ns(e["ts"])))
            elif e["ph"] == "f" and pid == pids["Ascend Hardware"]:
                destinations[e["tid"], ns(e["ts"])].append(e["id"])
        if e["ph"] != "X":
            continue
        s = Span.read(e)
        if name.startswith("ProfilerStep#"):
            steps[int(name.split("#")[-1])] = s
        if cat == "cpu_op" and name.startswith("aten::"):
            cpu.append(s)
        elif cat in ("enqueue", "dequeue"):
            coll = enq if cat == "enqueue" else deq
            cid = s.args["correlation_id"]
            if cid in coll:
                raise ValueError("duplicate queue id")
            coll[cid] = s
        if pid == pids["CANN"]:
            anchors[s.tid, s.start, s.args.get("connection_id")].append(s)
            if name.startswith(("AscendCL@aclrtLaunchKernel", "AscendCL@aclrtMemcpy")):
                api.append(s)
            elif name in ("AscendCL@aclrtSynchronizeDeviceWithTimeout", "AscendCL@aclrtSynchronizeDevice"):
                syncs.append(s)
            elif name.endswith("GetWorkspaceSize"):
                ws.append(s)
        elif pid == pids["Ascend Hardware"] and is_work(s.args.get("Task Type", "")):
            work.append(s)
    api_index = Lanes(api)
    matches = {}
    for task in work:
        ids = destinations[task.tid, task.start]
        if len(ids) != 1 or len(sources[ids[0]]) != 1:
            continue
        tid, start = sources[ids[0]][0]
        candidates = anchors[tid, start, task.args["connection_id"]]
        if len(candidates) != 1:
            continue
        calls = api_index.children(candidates[0])
        if len(calls) == 1:
            matches[id(task)] = (candidates[0], calls[0])
    syncs.sort(key=lambda s:s.start)
    sync_starts = [s.start for s in syncs]
    lower, upper = [], []
    for task in work:
        if id(task) not in matches:
            continue
        _, call = matches[id(task)]
        # A real device task cannot start before the Host enters the issuing API.
        lower.append((call.start-task.start, task, call))
        i = bisect_right(sync_starts, call.end)
        if i < len(syncs):
            # A device-wide synchronize after the API returns must see this work.
            upper.append((syncs[i].end-task.end, task, syncs[i]))
    def witness(item):
        shift, task, host = item
        return dict(shift_ns=shift, task=task.name, stream=task.tid, task_id=task.args["Task Id"],
                    task_start_ns=task.start, task_end_ns=task.end, host=host.name,
                    host_start_ns=host.start, host_end_ns=host.end)
    low = max(lower, key=lambda r:r[0])
    high = min(upper, key=lambda r:r[0])
    bounds = dict(lower=witness(low), upper=witness(high), launches=len(lower),
                  launch_then_global_sync=len(upper), sync_count=len(syncs), work=len(work),
                  trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
    return bounds, work, matches, steps, enq, deq, enclosing_aten(cpu,enq.values()), Lanes(ws)


def common_extrema(items, low, high):
    """Extrema of summed overlaps, with one clock shift shared by all rows."""
    value=0
    slope=0
    changes=defaultdict(int)
    for left,right,a,b in items:
        value+=max(0,min(right+low,b)-max(left+low,a))
        for edge,delta in ((a-right,1),(b-right,-1),(a-left,-1),(b-left,1)):
            if edge<=low:
                slope+=delta
            elif edge<high:
                changes[edge]+=delta
    last=low
    smallest=largest=value
    for edge in sorted(changes):
        value+=(edge-last)*slope
        smallest,largest=min(smallest,value),max(largest,value)
        slope+=changes[edge]
        last=edge
    value+=(high-last)*slope
    return min(smallest,value),max(largest,value)


def attribute(leg, output, selected_steps):
    bounds, work, matches, steps, enq, deq, owners, workspace = join(leg)
    output.mkdir(parents=True, exist_ok=True)
    (output/"bounds.json").write_text(json.dumps(bounds, indent=2))
    low, high = bounds["lower"]["shift_ns"]-1, bounds["upper"]["shift_ns"]+1
    if low > high:
        raise ValueError("no common clock offset satisfies the observed causal constraints")
    deq_index = Lanes(deq.values())
    by_start = defaultdict(list)
    for task in work:
        by_start[task.start].append(task)
    end = None
    rows = []
    parts=defaultdict(list)
    for t, group in sorted(by_start.items()):
        left, end = end, max(end or 0,max(x.end for x in group))
        if left is None or left >= t:
            continue
        step = next((n for n,s in steps.items() if s.start <= left+low and t+high <= s.end),None)
        if step is None or step not in selected_steps:
            continue
        row = dict(step=step, gap_ns=t-left, device_begin_ns=left, device_end_ns=t, state="ambiguous")
        rows.append(row)
        if len(group) != 1:
            continue
        task, = group
        row.update(task=task.name, stream=task.tid, task_id=task.args["Task Id"])
        if id(task) not in matches:
            continue
        anchor, call = matches[id(task)]
        consumer = deq_index.parent(anchor)
        if consumer is None:
            continue
        cid = consumer.args["correlation_id"]
        producer, owner = enq.get(cid), owners.get(cid)
        if (producer is None or producer.name.split("@",1)[-1]!=consumer.name.split("@",1)[-1]
                or producer.start > consumer.start or not steps[step].start <= producer.start < steps[step].end):
            continue
        op = producer.name.split("@",1)[-1]
        row.update(state="linked", queue_id=cid, op=op)
        # Each named interval shares the same unknown device-to-Host shift.
        lo = left+low
        edges = []
        if owner is not None:
            edges.append(("before_aten",owner.start))
            w = [s for s in workspace.children(owner) if s.end<=producer.start and
                 s.name=="AscendCL@"+op+"GetWorkspaceSize"]
            if len(w)==1:
                edges.extend((("producer_before_workspace",w[0].start),("workspace",w[0].end)))
            edges.append(("producer_to_enqueue",producer.start))
        else:
            edges.append(("before_enqueue_unresolved",producer.start))
        edges.extend((("enqueue",min(producer.end,consumer.start)),("queue_wait",consumer.start),
                      ("consumer_before_acl",call.start),("acl",call.end),("after_acl",t+high)))
        for name, stop in edges:
            a,b = lo,max(lo,stop)
            # Overlap is piecewise linear with breakpoints when two edges meet.
            shifts = [low,high]+[v for v in (a-left,b-left,a-t,b-t) if low<v<high]
            nsvals = [max(0,min(t+k,b)-max(left+k,a)) for k in shifts]
            row[name+"_min_ns"], row[name+"_max_ns"] = min(nsvals),max(nsvals)
            for group_name in ("ALL",op):
                parts[step,group_name,name].append((left,t,a,b))
            lo=b
    write_csv(output/"bounded_gaps.csv.gz",rows,compressed=True)
    totals = []
    for step in sorted(set(steps)&set(selected_steps)):
        selected = [r for r in rows if r["step"]==step]
        for op in sorted({"ALL"} | {r.get("op","unlinked") for r in selected}):
            subset = [r for r in selected if op=="ALL" or r.get("op","unlinked")==op]
            for field in ["gap_ns"]+sorted({k for r in subset for k in r if k.endswith(("_min_ns","_max_ns"))}):
                totals.append(dict(step=step, op=op, field=field, **stats([r.get(field,0) for r in subset])))
    write_csv(output/"summary.csv",totals)
    common=[]
    for (step,op,name),items in sorted(parts.items()):
        l,h=common_extrema(items,low,high)
        if not 0<=l<=h<=sum(b-a for a,b,_,_ in items):
            raise ValueError("common-offset overlap is outside the measured duration")
        common.append(dict(step=step,op=op,phase=name,observations=len(items),
                           lower_ms=l/1e6,upper_ms=h/1e6))
    write_csv(output/"common_shift_bounds.csv",common)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--steps", type=int, nargs="+", default=[4,5])
    args = parser.parse_args()
    attribute(args.leg, args.out, args.steps)
