"""Join individual PTA copy calls to physical SDMA and partition the preceding device gap."""
import bisect
import csv
import gzip
import hashlib
import importlib.util
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

OUT = Path(__file__).resolve().parent
RUNS = OUT.parents[1] / "runs"
CASES = {
    "original_ab": "20261119T135151Z-ab-l1-xe0mvd6a/01-A",
    "original_ba": "20261119T140634Z-ba-l1-1jllivx5/02-A",
    "overflow01_ab": "20261123T132159Z-ab-l1-iicxxhbo/01-A",
    "overflow01_ba": "20261123T132015Z-ba-l1-rdq9o84e/02-A",
}
# Use the reader archived with the measured runner. Trace timestamps are strings in us.
reader_path = (RUNS / CASES["original_ab"]).parent / "inputs" / "analyze_trace.py"
if not reader_path.is_file():
    reader_path = OUT / "analyze_trace.py"
spec = importlib.util.spec_from_file_location("npu_trace", reader_path)
reader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reader)


@dataclass(slots=True)
class Span:
    start: int
    end: int
    tid: int
    pid: int
    name: str
    key: int | None

    @classmethod
    def parse(cls, event):
        args = event.get("args", {})
        start = reader.ns(event["ts"])
        return cls(start, start + reader.ns(event.get("dur", 0)), event["tid"],
                   event["pid"], event["name"], args.get("connection_id",
                                                         args.get("correlation_id")))


class LaneIndex:
    """Non-nested spans of a given kind, indexed by OS thread."""

    def __init__(self, events):
        self.rows = defaultdict(list)
        for e in events:
            self.rows[e.tid].append(e)
        self.starts = {}
        for tid, rows in self.rows.items():
            rows.sort(key=lambda e: e.start)
            self.starts[tid] = [e.start for e in rows]

    def inside(self, parent):
        starts = self.starts.get(parent.tid, [])
        end = bisect.bisect_left(starts, parent.end)
        start = bisect.bisect_left(starts, parent.start)
        return [e for e in self.rows[parent.tid][start:end] if e.end <= parent.end]

    def owner(self, child):
        starts = self.starts.get(child.tid, [])
        i = bisect.bisect_right(starts, child.start) - 1
        if i >= 0:
            span = self.rows[child.tid][i]
            if child.end <= span.end:
                return span
        return None


CPU_COPY = "aten::copy_"
ACL = "AscendCL@aclnnInplaceCopy"
WS = ACL + "GetWorkspaceSize"
MEM = "AscendCL@aclrtMemcpyAsync"
PHASES = ["before_copy", "copy_before_ws", "workspace_api", "copy_to_enqueue",
          "enqueue_mark", "queue_wait", "dequeue_before_memcpy", "memcpy_api",
          "after_memcpy"]


def prior_gaps(tasks):
    """Only internal gaps. Group simultaneous starts so ties are never counted twice."""
    by_start = defaultdict(list)
    for t in tasks:
        by_start[t.start].append(t)
    gaps = {}
    frontier = None
    for start, rows in sorted(by_start.items()):
        if frontier is not None and start > frontier and len(rows) == 1:
            gaps[(rows[0].tid, start)] = frontier
        frontier = max(frontier or 0, max(t.end for t in rows))
    return gaps


def copy_parents(ops):
    parents = {}
    for tid, spans in ops.items():
        stack = []
        for event in sorted(spans, key=lambda e: (e.start, -e.end)):
            while stack and stack[-1].end < event.end:
                stack.pop()
            if event.name == CPU_COPY:
                ancestors = [s.name for s in stack if s.end > event.start]
                parents[(tid,event.start)] = "/".join(ancestors[-3:])
            stack.append(event)
    return parents


def consecutive_runs(rows, chronological, key_of, field):
    matched = {key_of(r):r for r in rows}
    chain = []
    for key in chronological + [None]:
        if key in matched:
            chain.append(matched[key])
        elif chain:
            for r in chain:
                r[field] = len(chain)
            chain.clear()


def partition(previous_end, device_start, cpu, ws, en, de, mem):
    # Consumer can begin before the enqueue end mark. Truncate that mark at dequeue start.
    until_dequeue = min(en.end, de.start)
    edges = [cpu.start, ws.start if ws else cpu.start, ws.end if ws else cpu.start,
             en.start, until_dequeue, de.start, mem.start, mem.end]
    if edges != sorted(edges):
        raise ValueError(f"unexpected host clock ordering for queue id {en.key}")
    left = previous_end
    parts = {}
    for name, boundary in zip(PHASES, edges + [device_start]):
        right = max(left, min(device_start, boundary))
        parts[name] = right - left
        left = right
    return parts


def analyze(label, relpath):
    path = RUNS / relpath / "trace_view.json.gz"
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    named = defaultdict(list)
    work, all_hw, dma, stage = [], [], [], []
    enqs, deqs, qflow, flow = {}, {}, defaultdict(set), defaultdict(set)
    process_names = {}
    kernel_launches = []
    all_dequeues = []
    cpu_ops = defaultdict(list)
    for e in reader.events(path):
        name, cat, ph = e["name"], e.get("cat"), e["ph"]
        args = e.get("args") or {}
        if ph == "M" and name == "process_name":
            process_names[e["pid"]] = args["name"]
        if ph in ("s", "f") and cat in ("HostToDevice", "async_task_queue"):
            target = flow if cat == "HostToDevice" else qflow
            key = (e["id"], ph)
            value = (e["pid"], e["tid"], reader.ns(e["ts"]))
            target[key].add(value)
        if ph != "X":
            continue
        if cat == "cpu_op":
            cpu_ops[e["tid"]].append(Span.parse(e))
        if cat == "dequeue":
            all_dequeues.append(Span.parse(e))
        if name in (CPU_COPY, ACL, WS, MEM) or name.startswith("ProfilerStep#"):
            named[name].append(Span.parse(e))
        if name.startswith("AscendCL@aclrtLaunchKernel"):
            kernel_launches.append(Span.parse(e))
        if name in reader.STAGES:
            stage.append(Span.parse(e))
        if name in ("Enqueue@aclnnInplaceCopy", "Dequeue@aclnnInplaceCopy"):
            span = Span.parse(e)
            dest = enqs if cat == "enqueue" else deqs
            if span.key in dest:
                raise ValueError(f"duplicate {cat} for {span.key}")
            dest[span.key] = span
        typ = args.get("Task Type")
        if typ and "Physic Stream Id" in args and "Task Id" in args and e.get("dur", 0) > 0:
            span = Span.parse(e)
            all_hw.append(span)
            if typ in reader.CORE or typ == "AI_CPU" or reader.DATA_TASK.match(typ):
                work.append(span)
            if typ == "SDMA_SQE" and name == "MEMCPY_ASYNC":
                dma.append(span)
    hw_pids = {e.pid for e in all_hw}
    if len(hw_pids) != 1 or "hardware" not in process_names[next(iter(hw_pids))].lower():
        raise ValueError("need one Ascend Hardware PID")
    g_work = prior_gaps(work)
    g_raw = prior_gaps(all_hw)
    dma_by_cid = defaultdict(list)
    for t in dma:
        dma_by_cid[t.key].append(t)
    f_dest = defaultdict(list)
    for (fid, side), locations in flow.items():
        if side == "f":
            device_locations = [loc for loc in locations if loc[0] in hw_pids]
            if len(device_locations) == 1:
                f_dest[device_locations[0]].append(fid)
    copy_index, acl_index, ws_index, mem_index = (LaneIndex(named[n]) for n in (CPU_COPY, ACL, WS, MEM))
    launch_index = LaneIndex(kernel_launches)
    parents = copy_parents(cpu_ops)
    rows, audit = [], []
    for k in (4, 5):
        (step,) = named[f"ProfilerStep#{k}"]
        counts = Counter()
        current = []
        for de in deqs.values():
            if not step.start <= de.start < de.end <= step.end:
                continue
            counts["copy_dequeues"] += 1
            mems = mem_index.inside(de)
            if len(mems) != 1 or launch_index.inside(de):
                continue
            counts["one_memcpy_no_launch"] += 1
            mem = mems[0]
            en = enqs.get(de.key)
            if en is None or qflow.get((de.key, "s")) != {(en.pid, en.tid, en.start)} or qflow.get((de.key, "f")) != {(de.pid, de.tid, de.start)}:
                counts["queue_unmatched"] += 1
                continue
            cpu = copy_index.owner(en)
            acls = acl_index.inside(de)
            if cpu is None or len(acls) != 1:
                counts["parent_unmatched"] += 1
                continue
            acl = acls[0]
            if not acl.start <= mem.start < mem.end <= acl.end:
                raise ValueError("memcpy API outside ACLNN")
            devices = dma_by_cid[acl.key]
            if len(devices) != 1:
                counts["device_unmatched"] += 1
                continue
            device = devices[0]
            ids = [fid for fid in f_dest[(device.pid, device.tid, device.start)]
                   if flow.get((fid, "s")) == {(acl.pid, acl.tid, acl.start)}]
            if len(ids) != 1:
                counts["flow_unmatched"] += 1
                continue
            counts["closed_chain"] += 1
            if not step.start <= device.start < step.end:
                counts["outside_step"] += 1
                continue
            ws_events = ws_index.inside(cpu)
            if len(ws_events) > 1:
                raise ValueError("ambiguous workspace scope")
            ws = next(iter(ws_events), None)
            raw_start = g_raw.get((device.tid, device.start), device.start)
            prev = g_work.get((device.tid, device.start), device.start)
            if min(raw_start, prev) < step.start:
                counts["gap_crosses_step"] += 1
                continue
            parts = partition(prev, device.start, cpu, ws, en, de, mem)
            scopes = [s for s in stage if s.start <= en.start < s.end]
            scope = min(scopes, key=lambda s: s.end - s.start).name if scopes else "outside_markers"
            current.append(dict(case=label, step=k, stage=reader.STAGES.get(scope, scope),
                             cpu_ancestors=parents.get((cpu.tid,cpu.start)),
                             queue_id=en.key, host_to_device_id=ids[0], aclnn_connection_id=acl.key,
                             memcpy_connection_id=mem.key, producer_tid=en.tid, consumer_tid=de.tid,
                             device_stream=device.tid, device_start_ns=device.start, prev_work_end_ns=prev,
                             workspace_seen=int(ws is not None), first_visible=next((n for n in PHASES if parts[n]), "no_gap"),
                             copy_preparation_us=(en.start - cpu.start)/1000,
                             enqueue_us=(en.end-en.start)/1000, wait_us=max(0,de.start-en.end)/1000,
                             service_us=(de.end-de.start)/1000, acl_memcpy_us=(mem.end-mem.start)/1000,
                             ws_us=(ws.end-ws.start)/1000 if ws else None,
                             gap_work_us=(device.start-prev)/1000, gap_all_raw_us=(device.start-raw_start)/1000,
                             **{f"exposed_{name}_us": t/1000 for name,t in parts.items()}))
        service_keys = [e.key for e in sorted(all_dequeues,key=lambda e:e.start) if step.start <= e.start < step.end]
        if len({e.tid for e in all_dequeues}) != 1:
            raise ValueError("multiple queue consumers; cannot use one service order")
        device_keys = [(e.tid,e.start) for e in sorted(work,key=lambda e:e.start) if step.start <= e.start < step.end]
        consecutive_runs(current, service_keys, lambda r:r["queue_id"], "consecutive_dequeues")
        consecutive_runs(current, device_keys, lambda r:(r["device_stream"],r["device_start_ns"]), "consecutive_work_tasks")
        rows.extend(current)
        audit.append(dict(case=label, path=relpath+"/trace_view.json.gz", sha256=sha, step=k,
                          **counts))
    return rows, audit


def summarize(rows, dimensions):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[d] for d in dimensions)].append(row)
    result = []
    for key, sample in sorted(groups.items()):
        r = dict(zip(dimensions,key))
        r.update(copies=len(sample), gaps=sum(s["gap_work_us"]>0 for s in sample))
        for metric in ("copy_preparation_us","ws_us","enqueue_us","wait_us","service_us","acl_memcpy_us","gap_work_us"):
            vals = sorted(s[metric] for s in sample if s[metric] is not None)
            for tail,f in (("p50",0.50),("p95",0.95),("p99",0.99),("max",1)):
                r[f"{metric}_{tail}"] = vals[int((len(vals)-1)*f)] if vals else None
        for metric in ["gap_work_us","gap_all_raw_us"] + ["exposed_"+n+"_us" for n in PHASES]:
            r[metric[:-3]+"_ms"] = round(sum(s[metric] for s in sample)/1000,6)
        result.append(r)
    return result


def write(name, rows, compressed=False):
    opener = gzip.open if compressed else open
    with opener(OUT/name, "wt", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    rows, audit = [], []
    for label, path in CASES.items():
        items, checks = analyze(label, path)
        rows += items
        audit += checks
        print(label, len(items), checks, flush=True)
    write("joins.csv", audit)
    write("copies.csv.gz", rows, compressed=True)
    write("steps.csv", summarize(rows, ("case","step")))
    write("by_stage.csv", summarize(rows, ("case","step","stage","workspace_seen")))
    write("by_device_sequence.csv", summarize(rows, ("case","step","consecutive_work_tasks")))
    write("by_cpu_parent.csv", summarize(rows, ("case","step","cpu_ancestors")))
    write("by_first_visible.csv", summarize(rows, ("case","step","first_visible")))
