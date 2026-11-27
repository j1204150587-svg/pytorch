"""Count real device submissions linked to stock/dispatched RMS scopes in a Level0 trace."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "causal_tril"))
from core_budget import intersect, merge, ns, read_leg


def count(trace, native=False):
    cpu, free, _ = read_leg(trace)
    with (gzip.open if trace.suffix == ".gz" else open)(trace, "rt", encoding="utf-8-sig") as f:
        doc = json.load(f)
    rows = doc["traceEvents"] if isinstance(doc, dict) else doc
    hardware_pid, = {r["pid"] for r in rows if r["name"] == "process_name" and r["args"]["name"] == "Ascend Hardware"}
    hw = defaultdict(list)
    for r in rows:
        if r["ph"] == "X" and r["pid"] == hardware_pid:
            hw[r["tid"], ns(r["ts"])].append(r)
    links = {}
    for r in rows:
        if r.get("cat") == "async_npu":
            links.setdefault(r["id"], {})[r["ph"]] = r
    tasks = []
    for endpoints in links.values():
        if set(endpoints) != {"s", "f"}:
            continue
        a, b = endpoints["s"], endpoints["f"]
        if b["pid"] != hardware_pid:
            continue
        matches = hw[b["tid"], ns(b["ts"])]
        if len(matches) != 1:
            raise ValueError(f"ambiguous device endpoint {b}")
        tasks.append((a["tid"], ns(a["ts"]), matches[0]))
    top = ("aten::pow", "aten::mean", "aten::add", "aten::rsqrt", "aten::mul", "aten::mul")
    output = []
    for beg, end, step, main, _ in cpu:
        if not step.startswith("ProfilerStep#"):
            continue
        rows = [r for r in cpu if beg <= r[0] and r[1] <= end]
        (fstart, fend, *_), = [r for r in rows if r[2] == "hstu_forward_and_loss"]
        (bstart, bend, *_), = [r for r in rows if r[2] == "deepspeed_backward"]
        fw = [r for r in rows if r[3] == main and fstart <= r[0] and r[1] <= fend and r[2].startswith("aten::")]
        level = []
        for r in sorted(fw, key=lambda v: (v[0], -v[1])):
            if not level or r[1] > level[-1][1]:
                level.append(r)
        groups = [level[i:i+len(top)] for i in range(len(level)) if tuple(r[2] for r in level[i:i+len(top)]) == top]
        sequences = set() if native else {r[4] for group in groups for r in group}
        parents = [r for r in rows if bstart <= r[0] and r[1] <= bend and r[2].startswith("autograd::engine::evaluate_function: ")]
        bwd = []
        for a, b, n, tid, seq in rows:
            if seq not in sequences or n.startswith("aten::") or not (bstart <= a and b <= bend):
                continue
            hits = [p for p in parents if p[2].endswith(": " + n) and p[3] == tid and p[0] <= a and b <= p[1]]
            if len(hits) != 1:
                raise ValueError(f"ambiguous backward parent {n} at {a}")
            bwd.append(hits[0])
        def stats(scopes):
            windows = merge([r[:2] for r in scopes])
            found = [dev for tid, t, dev in tasks if any(r[3] == tid and r[0] <= t < r[1] for r in scopes)]
            streams = Counter(f'{dev["args"]["Physic Stream Id"]}/{dev["args"]["Task Type"]}' for dev in found)
            devices = Counter(dev["name"] for dev in found)
            return {"host_scopes":len(scopes), "host_union_ms": sum(b-a for a,b in windows)/1e6,
                    "free_intersection_ms": intersect(windows, free)/1e6, "device_tasks":len(found),
                    "by_stream_task_type":dict(streams), "by_device_name":dict(devices)}
        if native:
            groups = [[r] for r in rows if fstart <= r[0] and r[1] <= fend and r[2] == "npu_hstu_native_rms::rms"]
            bwd = [r for r in parents if "CppNode<(anonymous namespace)::NativeRowRms>" in r[2]]
            if len(groups) != 9 or len(bwd) != 9:
                raise ValueError("expected nine native forward and backward RMS scopes")
        output.append({"step": step, "rms_instances": len(groups), "fwd": stats([r for g in groups for r in g]),
                       "bwd": stats(bwd)})
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("out", type=Path)
    args = parser.parse_args()
    report = count(args.trace)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    for row in report:
        print(row["step"], row["rms_instances"], [(name, row[name]["host_scopes"], row[name]["device_tasks"],
              row[name]["free_intersection_ms"]) for name in ("fwd", "bwd")])
