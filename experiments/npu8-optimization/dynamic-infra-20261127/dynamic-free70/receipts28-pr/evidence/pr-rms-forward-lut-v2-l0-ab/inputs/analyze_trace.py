"""Measure observed Ascend device work and host-side DeepSpeed stages in trace_view.json."""
import argparse
from bisect import bisect_right
from collections import Counter, defaultdict
from decimal import Decimal
import gzip
import json
from pathlib import Path
import re


STAGES = {
    "input_generation_and_h2d": "input",
    "hstu_forward_and_loss": "forward",
    "deepspeed_backward": "backward",
    "optimizer_step_including_native_deepspeed_norm": "optimizer",
    "deepspeed_native_get_grad_norm_direct": "native_norm",
}
CORE = {"AI_CORE", "AICORE", "AI_VECTOR_CORE", "VECTOR_CORE", "AIC", "AIV",
        "MIX_AIC", "MIX_AIV", "MIX_AICORE", "MIX_VECTOR_CORE", "AI_CORE_MIX",
        "MIX_AIC_AIV", "MIX_AIV_AIC", "KERNEL_AIVEC", "KERNEL_AICORE",
        "KERNEL_MIX_AIV", "KERNEL_MIX_AIC"}
NORM = {"aten::linalg_vector_norm", "aten::linalg_norm", "aten::norm",
        "aten::_foreach_norm", "aten::_fused_adam_norm"}
DATA_TASK = re.compile(r"^(?:(?:PCIE_|HCCS_)?DMA|SDMA|RDMA|HDMA|MEMCPY|MEMSET)(?:_|$)")


def events(path):
    """Stream the top-level event array without retaining CPU call stacks."""
    opener = gzip.open if path.suffix == ".gz" else open
    decoder = json.JSONDecoder()
    with opener(path, "rt", encoding="utf-8-sig") as source:
        buf = source.read(1 << 20)
        pos = len(buf) - len(buf.lstrip())
        if buf[pos:pos + 1] == "{":
            matched = re.search(r'"traceEvents"\s*:\s*\[', buf)
            while not matched:
                chunk = source.read(1 << 20)
                if not chunk:
                    raise ValueError("traceEvents array missing")
                buf += chunk
                matched = re.search(r'"traceEvents"\s*:\s*\[', buf)
            pos = matched.end()
        elif buf[pos:pos + 1] == "[":
            pos += 1
        else:
            raise ValueError("expected JSON array or Chrome traceEvents object")
        while True:
            while pos < len(buf) and buf[pos] in " \t\r\n,":
                pos += 1
            if buf[pos:pos + 1] == "]":
                return
            if pos < len(buf):
                try:
                    item, pos = decoder.raw_decode(buf, pos)
                except json.JSONDecodeError:
                    pass
                else:
                    yield item
                    continue
            chunk = source.read(1 << 20)
            if not chunk:
                raise ValueError("incomplete or invalid trace event")
            buf = buf[pos:] + chunk
            pos = 0


def ns(value):
    return int((Decimal(str(value).strip()) * 1000).to_integral_value())


def ms(value):
    return round(value / 1e6, 6)


def distribution(values):
    vals = sorted(values)
    if not vals:
        return {"count": 0}
    def q(fraction):
        return ms(vals[round((len(vals) - 1) * fraction)])
    return {"count": len(vals), "sum_ms": ms(sum(vals)), "median_ms": q(.5),
            "p90_ms": q(.9), "p99_ms": q(.99), "max_ms": ms(vals[-1])}


def merged(intervals):
    result = []
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(result[-1][1], b))
        else:
            result.append((a, b))
    return result


class Coverage:
    def __init__(self, intervals):
        self.spans = merged(intervals)
        self.ends = [b for a, b in self.spans]

    def window(self, a, b):
        cursor = a
        gaps = []
        for i in range(bisect_right(self.ends, a), len(self.spans)):
            left, right = self.spans[i]
            if left >= b:
                break
            if left > cursor:
                gaps.append((cursor, left))
            cursor = min(b, right)
        if cursor < b:
            gaps.append((cursor, b))
        elapsed = b - a
        uncovered = sum(y - x for x, y in gaps)
        return {"window_ms": ms(elapsed), "covered_ms": ms(elapsed - uncovered),
                "uncovered_ms": ms(uncovered),
                "uncovered_pct": round(100 * uncovered / elapsed, 3) if elapsed else None,
                "gaps": distribution([y - x for x, y in gaps]),
                "largest_gaps_ms_from_window_start": [[ms(x-a), ms(y-a)] for x, y in
                    sorted(gaps, key=lambda gap: gap[1]-gap[0], reverse=True)[:5]]}


def intra_thread_spacing(rows):
    per_lane = defaultdict(list)
    for start, end, lane, name in rows:
        per_lane[lane].append(start)
    diffs = []
    for starts in per_lane.values():
        starts.sort()
        diffs.extend(b - a for a, b in zip(starts, starts[1:]))
    return distribution(diffs)


def containment_index(markers):
    by_lane = defaultdict(list)
    for a, b, lane, _ in markers:
        by_lane[lane].append((a, b))
    index = {}
    for lane, rows in by_lane.items():
        starts, ends = [], []
        for a, b in sorted(rows):
            starts.append(a)
            ends.append(max(b, ends[-1]) if ends else b)
        index[lane] = starts, ends
    return index


def contains(index, lane, t):
    if lane not in index:
        return False
    starts, ends = index[lane]
    j = bisect_right(starts, t) - 1
    return j >= 0 and t < ends[j]


def analyze(path):
    pids, tids = {}, {}
    groups, types, phases = Counter(), Counter(), Counter()
    raw_tasks, labels, steps, norms, autograd = [], [], [], [], []
    flow_start, flow_end = {}, {}
    duplicate_flows = set()
    launched = []
    enqueues, dequeues = {}, {}
    for event in events(path):
        name = str(event.get("name", ""))
        ph, cat = event.get("ph"), str(event.get("cat", ""))
        lane = (event.get("pid"), event.get("tid"))
        args = event.get("args") or {}
        phases[ph] += 1
        if ph == "M":
            if name == "process_name":
                pids[lane[0]] = args.get("name", "")
            if name == "thread_name":
                tids[lane] = args.get("name", "")
            continue
        if ph in ("s", "f"):
            if name.lower() == "torch_to_npu" or cat.lower() == "hosttodevice":
                flow = ("torch" if name.lower() == "torch_to_npu" else "cann",
                        str(event.get("id", event.get("id2"))))
                endpoint = (lane, ns(event["ts"]))
                points = flow_start if ph == "s" else flow_end
                if flow in points and points[flow] != endpoint:
                    duplicate_flows.add(flow)
                points[flow] = endpoint
            continue
        if ph != "X":
            continue
        groups[(lane[0], cat)] += 1
        if cat in {"enqueue", "dequeue"} and "correlation_id" in args:
            queue_id = (lane[0], args["correlation_id"])
            begin = ns(event["ts"])
            target = enqueues if cat == "enqueue" else dequeues
            target[queue_id] = (begin, begin + ns(event.get("dur", 0)), lane, name)
            continue
        task_type = args.get("Task Type", args.get("task_type", args.get("taskType")))
        base_name = name.split(".", 1)[0]
        is_autograd = name.startswith("autograd::engine::evaluate_function:")
        is_launch = bool(re.search(r"(?:Runtime@rt|AscendCL@aclrt|^rt|^aclrt)(?:LaunchKernel|KernelLaunch)", name, re.I))
        is_step = name.startswith("ProfilerStep#")
        if not (task_type or name in STAGES or is_step or base_name in NORM or is_launch or is_autograd):
            continue
        a = ns(event["ts"])
        b = a + ns(event.get("dur", 0))
        if task_type:
            normalized = re.sub(r"[\s-]+", "_", str(task_type).strip().upper())
            stream = args.get("Physic Stream Id", args.get("Stream Id", args.get("Stream ID", args.get("stream_id"))))
            task_id = args.get("Task Id", args.get("Task ID", args.get("task_id")))
            raw_tasks.append((a, b, lane, name, normalized, stream is not None and task_id is not None))
            types[(lane[0], normalized)] += 1
        if name in STAGES:
            labels.append((a, b, lane, STAGES[name]))
        if is_step:
            steps.append((a, b, lane, name))
        if base_name in NORM:
            norms.append((a, b, lane, name))
        if is_autograd:
            autograd.append((a, b, lane, name))
        if is_launch:
            launched.append((a, b, lane, name))

    # msprof emits stream/task identifiers on raw task rows. When both are absent,
    # accept only explicit hardware stream tracks, excluding auxiliary annotations.
    hardware = [t for t in raw_tasks if t[1] > t[0] and
                (t[5] or ("hardware" in str(pids.get(t[2][0], "")).lower()
                           and re.match(r"^stream\s*\d+", str(tids.get(t[2], "")), re.I)))]
    core = [t for t in hardware if t[4] in CORE]
    cores_by_pid = defaultdict(list)
    hardware_by_pid = defaultdict(list)
    for t in core:
        cores_by_pid[t[2][0]].append(t)
    for t in hardware:
        hardware_by_pid[t[2][0]].append(t)
    labels.sort()
    steps.sort()
    norms.sort()
    # Workload step fallback applies if profiler does not emit ProfilerStep# markers.
    if not steps:
        for index, r in enumerate(x for x in labels if x[3] == "input"):
            next_start = min((s[0] for s in labels if s[3] == "input" and s[0] > r[0]), default=float("inf"))
            ends = [s[1] for s in labels if s[3] == "optimizer" and r[0] <= s[0] < next_start]
            if ends:
                steps.append((r[0], max(ends), r[2], f"input-to-optimizer#{index}"))
    all_cover = {p: Coverage((t[0], t[1]) for t in rows) for p, rows in hardware_by_pid.items()}
    work_cover = {
        p: Coverage((t[0], t[1]) for t in rows
                    if t[4] in CORE or t[4] in {"AI_CPU", "AICPU"} or DATA_TASK.match(t[4]))
        for p, rows in hardware_by_pid.items()
    }
    core_cover = {p: Coverage((t[0], t[1]) for t in rows) for p, rows in cores_by_pid.items()}
    dest = defaultdict(list)
    starts_by_pid = defaultdict(lambda: defaultdict(list))
    for i, task in enumerate(core):
        dest[(task[2], task[0])].append(i)
        starts_by_pid[task[2][0]][task[0]].append(i)
    source = {}
    cann_source = {}
    for kind in ("torch", "cann"):
        for flow, endpoint in flow_end.items():
            if flow[0] != kind or flow in duplicate_flows or flow not in flow_start:
                continue
            ids = dest.get(endpoint, [])
            if len(ids) == 1:
                source.setdefault(ids[0], (kind, *flow_start[flow]))
                if kind == "cann":
                    cann_source[ids[0]] = flow_start[flow][1]

    # Strict same-thread containment for correlated host sources. CANN service
    # threads can differ from the Python launch thread and remain unattributed.
    norm_index = containment_index(norms)
    worker_index = containment_index(autograd)

    def dispatch_gaps(pid, ids, a, b):
        # Only internal gaps in the full device-compute union ending at a
        # correlated kernel. Some profiler levels omit the host API duration.
        # Treat its flow source only as a CANN timestamp, with clock error.
        spans = core_cover[pid].spans
        first = bisect_right(core_cover[pid].ends, a)
        gaps, linked, before, after = [], [], [], []
        for j in range(max(1, first + 1), len(spans)):
            last_end, next_start = spans[j-1][1], spans[j][0]
            if next_start >= b:
                break
            following = starts_by_pid[pid][next_start]
            if not all(i in ids for i in following):
                continue
            gap = next_start - last_end
            gaps.append(gap)
            if not all(i in cann_source for i in following):
                continue
            host_start = min(cann_source[i] for i in following)
            if host_start > next_start:
                continue
            pre = max(0, host_start - last_end)
            linked.append(gap)
            before.append(pre)
            after.append(gap - pre)
        return {"global_internal_gaps_before_attributed_kernel": distribution(gaps),
                "gaps_with_cann_launch": distribution(linked),
                "gap_before_host_dispatch_start": distribution(before),
                "gap_after_host_dispatch_start": distribution(after)}

    def attributed(tasks):
        by_pid = defaultdict(list)
        for i in tasks:
            t = core[i]
            by_pid[t[2][0]].append(i)
        result = {}
        for p, ids in by_pid.items():
            rr = [core[i] for i in ids]
            cover = Coverage((r[0], r[1]) for r in rr)
            a, b = cover.spans[0][0], cover.spans[-1][1]
            result[str(p)] = {"count": len(rr), "names": Counter(r[3] for r in rr).most_common(12),
                              "span": cover.window(a, b),
                              "task_duration": distribution([r[1]-r[0] for r in rr]),
                              "device_start_spacing_within_stream": intra_thread_spacing([r[:4] for r in rr]),
                              "all_compute_during_span": core_cover[p].window(a, b),
                              "dispatch_evidence": dispatch_gaps(p, set(ids), a, b)}
        return result

    # Both queue endpoints are captured on the host. Require globally adjacent
    # dequeues on the same worker; another producer's intervening job breaks a gap.
    queue_previous = {}
    last_on_worker = {}
    for queue_id, (_, _, lane, _) in sorted(dequeues.items(), key=lambda kv: kv[1][0]):
        queue_previous[queue_id] = last_on_worker.get(lane)
        last_on_worker[lane] = queue_id

    def queue_timing(a, b, lane):
        ids = {i for i, (t, _, l, _) in enqueues.items() if a <= t < b and l == lane and i in dequeues}
        worker_gaps, pre_enqueue = [], []
        norms_dequeued = []
        by_worker = defaultdict(list)
        for i in ids:
            start, end, worker, name = dequeues[i]
            by_worker[worker].append((start, end))
            if name == "Dequeue@aclnnLinalgVectorNorm":
                norms_dequeued.append(dequeues[i])
            previous = queue_previous[i]
            if previous in ids:
                prev_end = dequeues[previous][1]
                worker_gaps.append(max(0, start - prev_end))
                pre_enqueue.append(max(0, min(start, enqueues[i][0]) - prev_end))
        return {
            "paired_requests": len(ids),
            "matched_worker_spans_ms": {str(k): ms(max(y for _, y in rows) - min(x for x, _ in rows))
                                        for k, rows in by_worker.items()},
            "dequeue_service": distribution([dequeues[i][1] - dequeues[i][0] for i in ids]),
            "adjacent_worker_gaps": distribution(worker_gaps),
            "worker_gap_before_next_enqueue_start": distribution(pre_enqueue),
            "enqueue_start_to_dequeue_start": distribution([dequeues[i][0] - enqueues[i][0] for i in ids]),
            "vector_norm_dequeue_start_spacing": intra_thread_spacing(norms_dequeued),
            "vector_norm_dequeue_service": distribution([v[1] - v[0] for v in norms_dequeued]),
        }

    def window_info(w):
        a, b, lane, label = w
        correl = [i for i, (_, slane, t) in source.items() if slane == lane and a <= t < b]
        norm_rows = [v for v in norms if v[2] == lane and a <= v[0] < b]
        canon = [v for v in norm_rows if v[3].split(".",1)[0] == "aten::linalg_vector_norm"]
        acl = [v for v in launched if a <= v[0] < b]
        details = {"name": label, "start_us": str(Decimal(a)/1000), "wall_ms": ms(b-a),
                "device_compute_in_host_window": {str(p): c.window(a, b) for p, c in core_cover.items()},
                "compute_aicpu_dma_in_host_window": {str(p): c.window(a,b) for p,c in work_cover.items()},
                "all_raw_device_tasks_in_host_window": {str(p): c.window(a,b) for p,c in all_cover.items()},
                "cpu_norm_op_counts": dict(Counter(r[3] for r in norm_rows)),
                "vector_norm_host_start_spacing": intra_thread_spacing(canon),
                "vector_norm_host_call_duration": distribution([v[1]-v[0] for v in canon]),
                "same_host_clock_queue": queue_timing(a, b, lane),
                "runtime_launch_event_counts_all_threads": dict(Counter(v[3] for v in acl)),
                "runtime_launch_spacing_within_thread": intra_thread_spacing(acl),
                "flow_correlated_compute": attributed(correl),
                "flow_correlated_compute_from_norm_calls": attributed(i for i in correl if contains(norm_index, source[i][1], source[i][2]))}
        if label == "backward":
            worker = [i for i, (kind, slane, t) in source.items()
                      if kind == "torch" and slane != lane and slane[0] == lane[0] and
                      a <= t < b and contains(worker_index, slane, t)]
            details["autograd_worker_flow_count"] = len(worker)
            details["flow_compute_including_autograd_workers"] = attributed(correl + worker)
        return details

    data = {"trace": str(path), "phases": dict(phases),
            "duration_event_tracks": [{"pid": p, "process": pids.get(p), "category": cat, "count": n}
                                      for (p,cat),n in groups.most_common()],
            "task_type_inventory": [{"pid": p, "process": pids.get(p), "type": typ, "count": n}
                                    for (p,typ),n in types.most_common()],
            "device_compute_types": sorted(CORE),
            "raw_task_count": len(hardware), "compute_task_count": len(core),
            "flow_matched_compute_count": dict(Counter(s[0] for s in source.values())),
            "cann_dispatch_flow_matched_compute_count": len(cann_source),
            "cann_source_later_than_device_kernel_start": sum(t > core[i][0] for i, t in cann_source.items()),
            "ambiguous_flow_ids": len(duplicate_flows),
            "method": "Chrome microseconds. Union per hardware PID across streams. Uncovered means no observed selected task. All-raw coverage may include waits. Flow attribution requires an exact hardware start and a source on the CPU marker thread. Norm-call attribution uses ATen norm intervals. Other work can fill a subset gap; all_compute_during_span checks this. dispatch_evidence splits at the CANN HostToDevice source timestamp, which may lack a visible API interval or precede/follow the physical launch because of clock alignment. Pairs with a source after the device start are omitted. same_host_clock_queue uses matched enqueue/dequeue IDs, attributing by enqueue time and thread. Gaps are between globally adjacent worker dequeues with both enqueues in the same marker. Host and device gaps can overlap.",
            "steps": []}
    for s in steps:
        row = window_info(s)
        row["stages"] = [window_info(l) for l in labels if l[2] == s[2] and s[0] <= l[0] < s[1]]
        data["steps"].append(row)
    if not steps:
        data["stages_without_step_markers"] = [window_info(l) for l in labels]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("-o", "--output", type=Path)
    opts = parser.parse_args()
    report = json.dumps(analyze(opts.trace), ensure_ascii=False, indent=2)
    if opts.output:
        opts.output.write_text(report + "\n", encoding="utf-8")
    else:
        print(report)
