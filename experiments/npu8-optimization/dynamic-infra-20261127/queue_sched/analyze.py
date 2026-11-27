"""Join target-thread scheduler state to linked PTA queue wait and consumer service gaps."""
import argparse
import ast
from bisect import bisect_right
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from shift_bounds import bound_rows


ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
sys.path.insert(0, str(ANALYSIS))
from analyze_dynamic_trace import Span, write_csv


def calibrate(directory, spans, info_root):
    limits = []
    for line in (directory / "clock.jsonl").read_text().splitlines():
        for name, before, inside_start, inside_end, after in json.loads(line):
            if name not in spans:
                continue
            s = spans[name]
            for endpoint, value, earliest, latest in (
                    ("start", s.start, before, inside_start), ("end", s.end, inside_end, after)):
                # Exporters may round epoch timestamps through floating point microseconds.
                limits.append(dict(marker=name, endpoint=endpoint, low=value-latest-500, high=value-earliest+500))
    if not limits:
        raise ValueError("no raw-clock calibration markers found in the active trace")
    low, high = max(r["low"] for r in limits), min(r["high"] for r in limits)
    if low > high:
        raise ValueError(f"raw clock and profiler anchors inconsistent by {low-high} ns")
    start_infos = []
    for path in info_root.rglob("start_info*"):
        if not path.is_file():
            continue
        try:
            obj = ast.literal_eval(path.read_text())
        except (SyntaxError, ValueError, OSError, UnicodeError):
            continue
        if isinstance(obj, dict) and {"collectionTimeBegin", "clockMonotonicRaw"} <= obj.keys():
            offset = int(Decimal(str(obj["collectionTimeBegin"])) * 1000) - int(obj["clockMonotonicRaw"])
            start_infos.append(dict(path=path.relative_to(info_root).as_posix(), offset_ns=offset,
                                    within_brackets=low <= offset <= high))
    # ProfilerPathManager.get_start_info_path prefers host/start_info. The
    # independent device_0/start_info.0 sample is a fallback only if it is absent.
    hosts = [info for info in start_infos if Path(info["path"]).name == "start_info"]
    selected = hosts or start_infos
    if len(selected) > 1 or any(not info["within_brackets"] for info in selected):
        raise ValueError(f"selected CANN clock source disagrees with runtime brackets: {selected}, {(low,high)}")
    return dict(raw_to_trace_offset_low_ns=low, raw_to_trace_offset_high_ns=high,
                uncertainty_width_ns=high-low, anchors=limits, cann_start_info=start_infos,
                exporter_start_info=selected[0] if selected else None)


def states_from_records(directory):
    with (directory / "scheduler.csv").open(newline="") as handle:
        rows = sorted(csv.DictReader(handle), key=lambda r: int(r["mono_raw_ns"]))
    intervals, current, previous_time, last_switch = [], None, None, None
    for row in rows:
        t, event, prev_state = int(row["mono_raw_ns"]), row["event"], row["prev_state"]
        if event == "wake" and current != "off_blocked":
            continue
        if event in {"in", "out"} and last_switch == event:
            raise ValueError("scheduler stream has two consecutive switches in the same direction")
        if current is not None and previous_time is not None and previous_time < t:
            label = current
            if current == "off_blocked" and event != "wake":
                label = "off_unknown_missing_wakeup"
            intervals.append((previous_time, t, label))
        if event == "in":
            current, last_switch = "scheduled", event
        elif event == "out":
            last_switch = event
            if prev_state == "R+":
                current = "runnable_after_preempt"
            elif prev_state == "R":
                current = "runnable_switch_out_R"
            elif prev_state.startswith(("S", "D", "I", "T", "t", "P")):
                current = "off_blocked"
            else:
                current = "off_unknown_prev_state"
        elif event == "wake":
            current = "runnable_after_wakeup"
        else:
            raise ValueError(f"unexpected scheduler transition {event}")
        previous_time = t
    return intervals


def overlaps(windows, ends, start, end):
    i = bisect_right(ends, start)
    while i < len(windows) and windows[i][0] < end:
        a, b, name = windows[i]
        yield max(start, a), min(end, b), name
        i += 1


def materialize(leg, result):
    # The host runner may keep the raw JSON in ASCEND_PROFILER_OUTPUT. Receipts
    # instead contain trace_view.json.gz in the leg and the same rank000 metrics.
    if (leg / "trace_view.json.gz").is_file():
        return leg
    files = list((leg / "rank000/profile").rglob("trace_view.json"))
    if len(files) != 1:
        raise ValueError(f"need one original trace, found {len(files)}")
    raw = result / "input"
    raw.mkdir(exist_ok=True)
    (raw / "rank000").mkdir(exist_ok=True)
    shutil.copyfile(leg / "rank000/metrics.json", raw / "rank000/metrics.json")
    with files[0].open("rb") as source, gzip.open(raw / "trace_view.json.gz", "wb", compresslevel=1) as output:
        shutil.copyfileobj(source, output)
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--capture", type=Path, help="separately copied rank000/queue_sched directory")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    directory = args.capture or args.leg / "rank000/queue_sched"
    meta = json.loads((directory / "scheduler.json").read_text())
    if meta["loss_before"] != meta["loss_after"] or meta["clock"] != "mono_raw":
        raise ValueError("unusable scheduler capture")
    leg = materialize(args.leg, args.out)
    joins = args.out / "links"
    source_path = leg / "trace_view.json.gz"
    with source_path.open("rb") as handle:
        source_digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            source_digest.update(chunk)
    summary = joins / "submissions.json"
    if not summary.exists() or json.loads(summary.read_text())["trace_sha256"] != source_digest.hexdigest():
        subprocess.run([sys.executable, "-B", str(ANALYSIS / "analyze_dynamic_trace.py"), str(leg), str(joins)], check=True)
    with gzip.open(source_path, "rt", encoding="utf-8-sig") as handle:
        trace = json.load(handle)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    scopes, anchors, services = {}, {}, []
    for event in events:
        if event["ph"] != "X":
            continue
        if event.get("cat") == "dequeue":
            if event["tid"] != meta["tid"] or event["pid"] != meta["pid"]:
                raise ValueError("filtered scheduler target differs from the actual PTA consumer")
            span = Span.read(event)
            services.append((span.start, span.end, "previous_dequeue"))
        elif event["name"].startswith("pta_sched_clock_"):
            anchors[event["name"]] = Span.read(event)
        elif event["name"].startswith("ProfilerStep#"):
            scopes[int(event["name"].split("#")[1])] = Span.read(event)
    del trace, events
    if not services or any(not any(n.startswith(f"pta_sched_clock_{step}_") for n in anchors) for step in scopes):
        raise ValueError("one active step lacks clock anchors or no PTA consumer observed")
    clock = calibrate(directory, anchors, args.leg / "rank000/profile")
    (args.out / "calibration.json").write_text(json.dumps(clock, indent=2))
    # Keep only interiors for which every feasible clock offset agrees on state.
    low, high = clock["raw_to_trace_offset_low_ns"], clock["raw_to_trace_offset_high_ns"]
    quantum = meta["trace_resolution_ns"]
    raw_states = states_from_records(directory)
    states = [(a + high + quantum, b + low - quantum, label)
              for a, b, label in raw_states if a + high + quantum < b + low - quantum]
    if not states:
        raise ValueError("no scheduler intervals resolved under calibrated uncertainty")
    ends = [b for _, b, _ in states]
    services.sort()
    if any(a[1] > b[0] for a,b in zip(services, services[1:])):
        raise ValueError("consumer records overlap on the target thread")
    service_ends = [b for _, b, _ in services]
    groups = defaultdict(Counter)
    sources = defaultdict(int)
    observed = defaultdict(list)

    def record(keys, phase, start, end):
        if end <= start:
            return
        split = Counter()
        for a, b, label in overlaps(states, ends, start, end):
            split[label] += b-a
        split["unresolved_clock_or_capture_edge"] = end-start-sum(split.values())
        for key in keys:
            groups[(*key,phase)].update(split)
            sources[(*key,phase)] += end-start
            if key[1] == "ALL":
                observed[key[0], phase].append((start,end))

    fields = ("before_aten_ns", "aten_before_workspace_ns", "workspace_ns", "aten_to_enqueue_remainder_ns",
              "before_enqueue_unresolved_ns", "enqueue_ns", "visible_queue_wait_ns", "consumer_before_api_ns",
              "acl_submission_api_ns", "after_api_ns")
    with gzip.open(joins / "device_gaps.csv.gz", "rt", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["state"] != "linked":
                continue
            step = int(row["step"])
            keys = [(step, "ALL", "ALL"), (step, row["op"], row["role"])]
            left = scopes[step].start + int(row["step_offset_ns"])
            cursor = left
            for field in fields:
                end = cursor + int(row.get(field) or 0)
                if field == "visible_queue_wait_ns":
                    p = cursor
                    for a, b, _ in overlaps(services, service_ends, cursor, end):
                        record(keys, "wait_outside_previous_dequeue", p, a)
                        record(keys, "wait_inside_previous_dequeue", a, b)
                        p = b
                    record(keys, "wait_outside_previous_dequeue", p, end)
                if field in {"consumer_before_api_ns", "acl_submission_api_ns"}:
                    record(keys, field, cursor, end)
                cursor = end
            if cursor-left != int(row["gap_ns"]):
                raise ValueError("device gap did not partition")
    rows = []
    for key, states in sorted(groups.items()):
        total = sources[key]
        if sum(states.values()) != total:
            raise ValueError("scheduler state totals differ from observable gap")
        step, op, role, phase = key
        rows.append(dict(step=step, op=op, role=role, phase=phase, total_ms=round(total/1e6,6),
                         **{k + "_ms": round(v/1e6,6) for k,v in states.items()}))
    write_csv(args.out / "exposed_scheduler.csv", rows)
    write_csv(args.out / "common_offset_bounds.csv", bound_rows(raw_states, observed, low, high, quantum))


if __name__ == "__main__":
    main()
