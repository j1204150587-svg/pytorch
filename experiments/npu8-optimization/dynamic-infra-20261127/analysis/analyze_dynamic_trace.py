"""Join dynamic HSTU PTA calls, CANN submissions, and preceding device gaps."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path


def ns(value):
    return int(Decimal(str(value)) * 1000)


@dataclass(slots=True)
class Span:
    start: int
    end: int
    tid: int
    name: str
    args: dict

    @classmethod
    def read(cls, event):
        start = ns(event["ts"])
        return cls(start, start + ns(event["dur"]), event["tid"], event["name"], event.get("args", {}))


class Lanes:
    """Index nonoverlapping intervals on each thread."""
    def __init__(self, spans):
        self.lanes = defaultdict(list)
        for span in spans:
            self.lanes[span.tid].append(span)
        self.starts = {}
        for tid, lane in self.lanes.items():
            lane.sort(key=lambda x: x.start)
            self.starts[tid] = [x.start for x in lane]

    def children(self, parent):
        starts = self.starts.get(parent.tid, ())
        lo = bisect_left(starts, parent.start)
        hi = bisect_left(starts, parent.end)
        return [x for x in self.lanes[parent.tid][lo:hi] if x.end <= parent.end]

    def parent(self, child):
        i = bisect_right(self.starts.get(child.tid, ()), child.start) - 1
        if i >= 0:
            parent = self.lanes[child.tid][i]
            if child.end <= parent.end:
                return parent
        return None


def enclosing_aten(cpu, enqueues):
    """Innermost same-thread ATen scope containing a complete enqueue."""
    marks = defaultdict(list)
    for span in cpu:
        marks[span.tid].append((span.start, 0, -span.end, span))
    for span in enqueues:
        marks[span.tid].append((span.start, 1, -span.end, span))
    owners = {}
    for lane in marks.values():
        stack = []
        for start, kind, _, span in sorted(lane, key=lambda v: v[:3]):
            while stack and stack[-1].end <= start:
                stack.pop()
            if kind == 0:
                stack.append(span)
            else:
                owner = next((x for x in reversed(stack) if span.end <= x.end), None)
                if owner:
                    owners[span.args["correlation_id"]] = owner
    return owners


def gap_partition(left, right, edges):
    cursor = left
    result = {}
    for field, endpoint in edges:
        end = max(cursor, min(right, endpoint))
        result[field] = end - cursor
        cursor = end
    if cursor != right:
        raise ValueError("gap partition failed to cover its interval")
    return result


def stats(values):
    samples = sorted(values)
    if not samples:
        return dict(n=0)
    return dict(n=len(samples), sum_ms=round(sum(samples) / 1e6, 6),
                **{label: round(samples[int(q * (len(samples) - 1))] / 1000, 3)
                   for label, q in (("p50_us", .5), ("p95_us", .95), ("p99_us", .99), ("max_us", 1.))})


def write_csv(path, rows, compressed=False):
    opener = gzip.open if compressed else open
    with opener(path, "wt", encoding="utf-8", newline="") as handle:
        if rows:
            fields = dict.fromkeys(key for row in rows for key in row)
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)


def is_work(task_type):
    core = {"AI_CORE", "AICORE", "AI_VECTOR_CORE", "VECTOR_CORE", "AIC", "AIV",
            "MIX_AIC", "MIX_AIV", "MIX_AICORE", "MIX_VECTOR_CORE", "AI_CORE_MIX",
            "MIX_AIC_AIV", "MIX_AIV_AIC", "AI_CPU", "AICPU"}
    return task_type in core or task_type.startswith(("KERNEL_", "SDMA", "HDMA", "RDMA",
                                                       "HCCS_DMA", "PCIE_DMA", "MEMCPY", "MEMSET"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg", type=Path, help="directory containing trace_view.json.gz and rank000/metrics.json")
    parser.add_argument("output", type=Path)
    parser.add_argument("--schedule", type=Path, help="scheduled shapes if per-step metrics have not been exported")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.leg / "trace_view.json.gz"
    with gzip.open(path, "rt", encoding="utf-8-sig") as handle:
        trace = json.load(handle)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    pids = {e["args"]["name"]: e["pid"] for e in events if e["name"] == "process_name"}
    if "CANN" not in pids:
        raise ValueError("a Level1 CANN timeline is required")
    metrics_path = args.leg / "rank000/metrics.json"
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_bytes())["steps"]
        shapes = {s["step"]: (s["batch_size"], s["seq_len"]) for s in metrics}
        shape_source = "rank000/metrics.json"
    elif args.schedule:
        shapes = {i: tuple(s) for i,s in enumerate(json.loads(args.schedule.read_bytes()))}
        shape_source = f"schedule only: {args.schedule.name}"
    else:
        raise ValueError("need metrics.json or --schedule")
    cpu, apis, work, all_hardware, workspace, syncs = [], [], [], [], [], []
    enqueues, dequeues, steps = {}, {}, {}
    src, dst, cann = (defaultdict(list) for _ in range(3))
    for event in events:
        ph, name, pid = event["ph"], event["name"], event["pid"]
        cat = event.get("cat")
        if cat == "HostToDevice":
            if ph == "s" and pid == pids["CANN"]:
                src[event["id"]].append((event["tid"], ns(event["ts"])))
            if ph == "f" and pid == pids["Ascend Hardware"]:
                dst[event["tid"], ns(event["ts"])].append(event["id"])
        if ph != "X":
            continue
        span = Span.read(event)
        if name.startswith("ProfilerStep#"):
            steps[int(name.split("#")[1])] = span
        if cat == "cpu_op" and name.startswith("aten::"):
            cpu.append(span)
        if cat in ("enqueue", "dequeue"):
            items = enqueues if cat == "enqueue" else dequeues
            key = span.args["correlation_id"]
            if key in items:
                raise ValueError(f"ambiguous {cat} ID {key}")
            items[key] = span
        if pid == pids["CANN"]:
            cann[span.tid, span.start, span.args.get("connection_id")].append(span)
            if name.endswith("GetWorkspaceSize"):
                workspace.append(span)
            if name.startswith(("AscendCL@aclrtLaunchKernel", "AscendCL@aclrtMemcpy")):
                apis.append(span)
            if name.startswith("AscendCL@aclrtSynchronize"):
                syncs.append(span)
        if pid == pids["Ascend Hardware"] and "Task Type" in span.args:
            all_hardware.append(span)
            if is_work(span.args["Task Type"]):
                work.append(span)
    del trace, events
    owners = enclosing_aten(cpu, enqueues.values())
    api_index, deq_index, ws_index = Lanes(apis), Lanes(dequeues.values()), Lanes(workspace)
    step_rows = sorted(steps.items(), key=lambda pair: pair[1].start)
    starts = [s.start for _, s in step_rows]

    def step_for(t):
        i = bisect_right(starts, t) - 1
        if i >= 0 and t < step_rows[i][1].end:
            return step_rows[i][0]
        return None

    def context(step):
        b, length = shapes[step]
        return dict(step=step, batch=b, seq_len=length,
                    first_step_with_input_shape=min(i for i, pair in shapes.items() if pair == (b, length)))

    ordinal_counts = Counter()
    ordinal_by_owner = {}
    for event in sorted(cpu, key=lambda x:(x.start, -x.end)):
        step=step_for(event.start)
        if step is not None:
            key=step,event.tid,event.name
            ordinal_by_owner[id(event)]=ordinal_counts[key]
            ordinal_counts[key]+=1

    all_calls, grouped, intervals = [], defaultdict(lambda: defaultdict(list)), {}
    for cid, enq in enqueues.items():
        step = step_for(enq.start)
        deq = dequeues.get(cid)
        if step is None or deq is None or deq.name.split("@",1)[-1] != enq.name.split("@",1)[-1]:
            continue
        op = enq.name.split("@", 1)[-1]
        owner = owners.get(cid)
        calls = api_index.children(deq)
        ws = [s for s in ws_index.children(owner) if s.end <= enq.start and s.name == "AscendCL@" + op + "GetWorkspaceSize"] if owner else []
        ws = ws[0] if len(ws) == 1 else None
        intervals[cid] = (enq, deq, owner, ws)
        atr = owner.name if owner else ""
        row = dict(**context(step), queue_id=cid, op=op, aten=atr,
                   role="main" if enq.tid == steps[step].tid else "worker",
                   producer_tid=enq.tid, consumer_tid=deq.tid,
                   step_offset_ns=enq.start-steps[step].start,
                   enqueue_ns=enq.end-enq.start, signed_queue_wait_ns=deq.start-enq.end,
                   dequeue_ns=deq.end-deq.start, launch_api_count=len(calls))
        if owner:
            row["aten_to_enqueue_ns"] = enq.start - owner.start
            row["aten_ordinal_in_thread"] = ordinal_by_owner.get(id(owner),"")
            if "Input Dims" in owner.args:
                row["aten_input_dims"] = json.dumps(owner.args["Input Dims"], separators=(",",":"))
        if ws:
            row["workspace_ns"] = ws.end - ws.start
        row["acl_api_sequence"] = "|".join(s.name for s in calls)
        row["acl_api_sum_ns"] = sum(s.end-s.start for s in calls)
        key = (step, op, atr, row["role"])
        for col in ("aten_to_enqueue_ns", "workspace_ns", "enqueue_ns", "signed_queue_wait_ns", "dequeue_ns"):
            if col in row:
                grouped[key][col].append(row[col])
        for api in calls:
            grouped[key][api.name.split("@")[-1]].append(api.end-api.start)
        all_calls.append(row)
    write_csv(args.output / "queue_calls.csv.gz", all_calls, compressed=True)
    write_csv(args.output / "operator_latency.csv", [
        dict(**context(step), op=op, aten=aten, role=role, metric=metric, **stats(values))
        for (step,op,aten,role), columns in sorted(grouped.items())
        for metric,values in sorted(columns.items())])

    gaps, gaps_by_op = [], defaultdict(lambda: defaultdict(list))
    frontier = None
    by_start = defaultdict(list)
    for task in work:
        by_start[task.start].append(task)
    for t, group in sorted(by_start.items()):
        left = frontier
        frontier = max(frontier or 0, max(s.end for s in group))
        step = step_for(t)
        if step is None or left is None or left >= t or left < steps[step].start:
            continue
        row = dict(**context(step), step_offset_ns=left-steps[step].start, gap_ns=t-left,
                   state="ambiguous_next_task")
        gaps.append(row)
        if len(group) != 1:
            continue
        task = group[0]
        row.update(task=task.name, stream=task.args["Physic Stream Id"], task_id=task.args.get("Task Id"),
                   state="unmatched_host_flow")
        ids = dst[task.tid, t]
        if len(ids) != 1 or len(src[ids[0]]) != 1:
            continue
        tid, source_time = src[ids[0]][0]
        anchors = cann[tid, source_time, task.args.get("connection_id")]
        if len(anchors) != 1:
            continue
        anchor = anchors[0]
        nested = api_index.children(anchor)
        deq = deq_index.parent(anchor)
        row.update(flow_id=ids[0], host_anchor=anchor.name, state="no_unique_consumer_submission")
        if len(nested) != 1 or deq is None:
            continue
        cid = deq.args["correlation_id"]
        if cid not in intervals:
            continue
        enq, deq, owner, ws = intervals[cid]
        launch = nested[0]
        if enq.start > deq.start or launch.start > t or step_for(enq.start) != step:
            row["state"] = "clock_or_step_boundary"
            continue
        op = enq.name.split("@", 1)[-1]
        role = "main" if enq.tid == steps[step].tid else "worker"
        edges = []
        if owner:
            edges.append(("before_aten_ns", owner.start))
            if ws:
                edges.extend((("aten_before_workspace_ns", ws.start), ("workspace_ns", ws.end)))
            edges.append(("aten_to_enqueue_remainder_ns", enq.start))
        else:
            edges.append(("before_enqueue_unresolved_ns", enq.start))
        edges.extend((("enqueue_ns", min(enq.end,deq.start)), ("visible_queue_wait_ns", deq.start),
                      ("consumer_before_api_ns",launch.start), ("acl_submission_api_ns",launch.end),
                      ("after_api_ns",t)))
        parts = gap_partition(left,t,edges)
        row.update(state="linked", op=op, role=role, queue_id=cid, **parts)
        row["first_visible"] = next(k for k,v in parts.items() if v)
        for name, duration in dict(gap_ns=t-left, **parts).items():
            gaps_by_op[step,op,role][name].append(duration)
    write_csv(args.output / "device_gaps.csv.gz", gaps, compressed=True)
    write_csv(args.output / "gap_attribution.csv", [
        dict(**context(step), op=op, role=role, metric=metric, **stats(values))
        for (step,op,role), columns in sorted(gaps_by_op.items()) for metric,values in sorted(columns.items())])

    per_step = []
    task_rows = []
    for step, scope in step_rows:
        def inside(span):
            return scope.start <= span.start < scope.end
        hardware = [x for x in all_hardware if inside(x)]
        lanes = defaultdict(list)
        for task in sorted(hardware, key=lambda x:(x.start, x.name, x.args.get("Task Id", -1))):
            lanes[task.args["Physic Stream Id"]].append((task.args["Task Type"],task.name))
        streams = {}
        for stream, seq in sorted(lanes.items()):
            def signature(items):
                return hashlib.sha256(json.dumps(items,separators=(",",":")).encode()).hexdigest()
            useful = [pair for pair in seq if is_work(pair[0])]
            streams[stream] = dict(all_n=len(seq), all_sha256=signature(seq),
                                   work_n=len(useful), work_sha256=signature(useful))
        slots=Counter()
        for task in sorted((x for x in hardware if is_work(x.args["Task Type"])),
                           key=lambda x:(x.start,x.name,x.args.get("Task Id",-1))):
            stream=task.args["Physic Stream Id"]
            row=dict(**context(step),stream=stream,work_sequence_sha256=streams[stream]["work_sha256"],
                     stream_work_slot=slots[stream],task=task.name,task_type=task.args["Task Type"],
                     task_id=task.args.get("Task Id"),queue_id="")
            slots[stream]+=1
            ids=dst[task.tid,task.start]
            if len(ids)==1 and len(src[ids[0]])==1:
                tid,source_time=src[ids[0]][0]
                anchors=cann[tid,source_time,task.args.get("connection_id")]
                if len(anchors)==1:
                    deq=deq_index.parent(anchors[0])
                    if deq and step_for(deq.start)==step:
                        row.update(queue_id=deq.args["correlation_id"],flow_id=ids[0],anchor=anchors[0].name)
            task_rows.append(row)
        ordered_aten = defaultdict(list)
        for event in sorted((x for x in cpu if inside(x)), key=lambda x:(x.start,-x.end,x.name)):
            ordered_aten[event.tid].append(event.name)
        aten_streams = [dict(role="main" if tid==scope.tid else "worker", n=len(seq),
                             sha256=hashlib.sha256(json.dumps(seq,separators=(",",":")).encode()).hexdigest())
                        for tid,seq in sorted(ordered_aten.items())]
        step_gaps = [r for r in gaps if r["step"]==step]
        per_step.append(dict(**context(step), hardware_types=Counter(x.args["Task Type"] for x in hardware),
                             acl_submission_apis=Counter(x.name for x in apis if inside(x)),
                             acl_stream_synchronizations=stats([x.end-x.start for x in syncs if inside(x)]),
                             aten_counts=Counter(x.name for x in cpu if inside(x)),
                             enqueue_counts=Counter(x.name for x in enqueues.values() if inside(x)),
                             aten_thread_sequences=aten_streams, streams=streams,
                             gap_states={s:stats([r["gap_ns"] for r in step_gaps if r["state"]==s])
                                 for s in sorted(set(r["state"] for r in step_gaps))},
                             largest_work_gaps=sorted(step_gaps,key=lambda r:-r["gap_ns"])[:12]))
    write_csv(args.output / "work_tasks.csv.gz",task_rows,compressed=True)
    with path.open("rb") as handle:
        sha = hashlib.file_digest(handle,"sha256").hexdigest()
    report = dict(trace_sha256=sha, shape_source=shape_source, quantile="floor(q * (n - 1))", steps=per_step)
    (args.output / "submissions.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__ == "__main__":
    main()
