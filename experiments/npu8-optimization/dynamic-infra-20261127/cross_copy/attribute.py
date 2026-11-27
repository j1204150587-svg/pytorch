"""Join #42/#44 Torch scopes to PTA queues, CANN memcpy, physical SDMA and preceding work gaps."""
import argparse
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
ANALYSIS = HERE.parent / "analysis"
sys.path.insert(0, str(ANALYSIS))
from analyze_dynamic_trace import Lanes, Span, stats, write_csv


MARKS = {"NPU42GuardedD2D", "NPU42MatchedD2D", "NPU44MatchedSourceD2D", "NPU44GuardedSourceD2D"}
PHASES = {"input_generation_and_h2d", "hstu_forward_and_loss", "deepspeed_backward",
          "optimizer_step_including_native_deepspeed_norm"}
FIELDS = ("aten_to_enqueue_ns", "enqueue_ns", "signed_queue_wait_ns", "dequeue_ns", "acl_api_sum_ns", "workspace_ns")


def rows(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def read_markers(trace_path):
    with gzip.open(trace_path, "rt", encoding="utf-8-sig") as stream:
        raw = json.load(stream)
    events = raw["traceEvents"] if isinstance(raw, dict) else raw
    intervals, requests, steps, phases, controls = [], [], [], [], []
    pids = {e["args"]["name"]: e["pid"] for e in events if e["name"] == "process_name"}
    for e in events:
        if e["ph"] != "X":
            continue
        name = e["name"]
        if name in MARKS:
            intervals.append(Span.read(e))
        elif name in PHASES:
            phases.append(Span.read(e))
        elif name.startswith("ProfilerStep#"):
            steps.append(Span.read(e))
        elif e.get("cat") == "enqueue":
            requests.append(Span.read(e))
        elif e["pid"] == pids["CANN"] and (name.startswith("AscendCL@aclrt") and "Event" in name):
            controls.append(Span.read(e))
    def step_at(t):
        return next((int(s.name.split("#")[-1]) for s in steps if s.start <= t < s.end), None)

    mark_index = Lanes(intervals)
    owners, qid_phase = {}, {}
    usage, marker_counts, ctrl_counts = Counter(), Counter(), Counter()
    for s in intervals:
        marker_counts[step_at(s.start), s.name] += 1
    for c in controls:
        ctrl_counts[step_at(c.start), c.name] += 1
    for q in requests:
        qid = str(q.args["correlation_id"])
        m = mark_index.parent(q)
        if m:
            owners[qid] = m.name
            usage[id(m)] += 1
        phase = next((s.name for s in phases if s.start <= q.start < s.end), "")
        qid_phase[qid] = phase
    multiplicities = Counter((step_at(m.start), m.name, usage[id(m)]) for m in intervals)
    return owners, qid_phase, marker_counts, multiplicities, ctrl_counts


def analyze_leg(variant, raw, common):
    owners, phases, markers, multiplicity, controls = read_markers(raw / "trace_view.json.gz")
    calls = rows(common / "queue_calls.csv.gz")
    tasks = defaultdict(list)
    for row in rows(common / "work_tasks.csv.gz"):
        if row["queue_id"]:
            tasks[row["step"], row["queue_id"]].append(row)
    gaps = defaultdict(list)
    for row in rows(common / "device_gaps.csv.gz"):
        if row.get("queue_id"):
            gaps[row["step"], row["queue_id"]].append(row)
    tagged = {}
    census = Counter()
    fallback = Counter()
    for q in calls:
        cid, step = q["queue_id"], q["step"]
        mark = owners.get(cid, "")
        if q["aten"] == "aten::copy_" and not mark:
            fallback[int(step), phases[cid], q["role"], q["op"]] += 1
        if not mark:
            continue
        key = step, cid
        census[int(step), mark, phases[cid], q["role"], q["op"]] += 1
        tagged[key] = {**q, "marker": mark, "phase": phases[cid], "tasks": tasks[key], "gaps": gaps[key]}
    metric = json.loads((raw / "rank000/metrics.json").read_bytes())["steps"]
    evidence = {
        "markers": [{"step":s,"shape":[metric[s]["batch_size"],metric[s]["seq_len"]],"name":m,"count":n}
                    for (s,m),n in sorted(markers.items(), key=str) if s is not None],
        "nested_enqueue_multiplicity": [{"step":s,"marker":m,"enqueues_per_marker":k,"markers":n}
                                        for (s,m,k),n in sorted(multiplicity.items(), key=str) if s is not None],
        "marked_queue_counts": [{"step":s,"marker":m,"phase":p,"role":r,"op":q,"count":n}
                                for (s,m,p,r,q),n in sorted(census.items())],
        "unmarked_aten_copy_queues": [{"step":s,"phase":p,"role":r,"op":o,"count":n}
                                     for (s,p,r,o),n in sorted(fallback.items())],
        "cann_event_API": [{"step":s,"name":c,"count":n} for (s,c),n in sorted(controls.items(),key=str)
                           if s is not None],
    }
    return tagged, evidence


def by_ordinal(records, marker):
    out = defaultdict(list)
    tids = defaultdict(set)
    for q in records.values():
        if q["marker"] == marker:
            g = int(q["step"]),q["role"]
            tids[g].add(q["producer_tid"])
            if q["aten"] == "aten::copy_" and q.get("aten_ordinal_in_thread", "") != "":
                out[(*g, int(q["aten_ordinal_in_thread"]))].append(q)
    if any(len(v)>1 for v in tids.values()):
        raise ValueError(f"{marker}: multiple producer threads per role")
    return out


def compare_cohort(x, y, category, rawrows, distributions, gaprows):
    summary = defaultdict(Counter)
    rawmetrics = defaultdict(list)
    gapmetrics = defaultdict(list)
    first_visible = defaultdict(Counter)
    for step, role, ordinal in sorted(x.keys() | y.keys()):
        key = step, role, ordinal
        rows_a, rows_b = x.get(key, []), y.get(key, [])
        counts = summary[step,role]
        counts["union_ordinals"] += 1
        if len(rows_a) != 1 or len(rows_b) != 1:
            counts["unmatched_or_multiple"] += 1
            continue
        a,b = rows_a[0], rows_b[0]
        counts["paired"] += 1
        def physical(q):
            return [(int(t["stream"]), int(t["stream_work_slot"]), t["task_type"], t["task"]) for t in q["tasks"]]
        p,q = physical(a),physical(b)
        valid = (len(p)==len(q)==1 and all(s[2].startswith("SDMA") for s in [p[0],q[0]]) and
                 a["acl_api_sequence"] == b["acl_api_sequence"] == "AscendCL@aclrtMemcpyAsync")
        counts["one_ACL_memcpy_one_physical_SDMA"] += int(valid)
        counts["also_same_physical_stream_and_work_slot"] += int(valid and p==q)
        for leg,c,devices in (("A",a,p),("B",b,q)):
            rawrows.append(dict(step=step,role=role,ordinal=ordinal,cohort=category,leg=leg,marker=c["marker"],
                                phase=c["phase"],queue_id=c["queue_id"],queue_op=c["op"],
                                acl=c["acl_api_sequence"],tasks=json.dumps(devices),**{k:c.get(k,"") for k in FIELDS}))
            for gap in c["gaps"]:
                first_visible[step,role,leg][gap.get("first_visible") or gap["state"]] += 1
            for scale in ("all", c["phase"] or "unscoped"):
                for f in FIELDS:
                    if c.get(f,"") != "":
                        rawmetrics[step,role,scale,leg,f].append(int(c[f]))
                for gap in c["gaps"]:
                    for f,v in gap.items():
                        if f.endswith("_ns") and f != "step_offset_ns" and v:
                            gapmetrics[step,role,scale,leg,f].append(int(v))
    for dest, table in ((distributions, rawmetrics), (gaprows, gapmetrics)):
        for (s,r,ph,leg,f),v in sorted(table.items()):
            dest.append(dict(cohort=category,step=s,role=r,phase=ph,leg=leg,metric=f,**stats(v)))
    return [{"cohort":category,"step":s,"role":r,**counts,
             "first_visible_A":first_visible[s,r,"A"], "first_visible_B":first_visible[s,r,"B"]}
            for (s,r),counts in sorted(summary.items())]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("pair", type=Path)
    cli.add_argument("out", type=Path)
    cli.add_argument("--reuse-common", action="store_true", help="reuse traces processed by compare_dynamic_legs.py")
    args = cli.parse_args()
    dirs = {v: next(args.pair.glob("*-" + v)) for v in ("A","B")}
    args.out.mkdir(parents=True,exist_ok=True)
    if not args.reuse_common:
        subprocess.run([sys.executable,"-B",str(ANALYSIS/"compare_dynamic_legs.py"),
                        str(dirs["A"]),str(dirs["B"]),str(args.out)],check=True)
    pairs = {v: analyze_leg(v,dirs[v],args.out/v) for v in ("A","B")}
    output = {"legs": {v:pairs[v][1] for v in pairs}, "cohorts":[]}
    raw, times, gaps = [], [], []
    for label, ma, mb in (("cross_source_44","NPU44MatchedSourceD2D","NPU44GuardedSourceD2D"),
                          ("same_stream_42","NPU42GuardedD2D","NPU42GuardedD2D")):
        output["cohorts"].extend(compare_cohort(by_ordinal(pairs["A"][0],ma),by_ordinal(pairs["B"][0],mb),
                                                label,raw,times,gaps))
    write_csv(args.out/"marked_copy_calls.csv.gz",raw,compressed=True)
    write_csv(args.out/"marked_copy_latency.csv",times)
    write_csv(args.out/"marked_copy_preceding_gaps.csv",gaps)
    (args.out/"copy_guard_evidence.json").write_text(json.dumps(output,indent=2) + "\n",encoding="utf-8")
    print(json.dumps(output["cohorts"]))


if __name__ == "__main__":
    main()
