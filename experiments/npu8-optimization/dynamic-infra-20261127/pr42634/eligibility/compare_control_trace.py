"""Compare device tasks inside the positive control's four synchronized markers."""
import argparse
from decimal import Decimal
import gzip
import json
from pathlib import Path


def ts(value):
    return int(Decimal(str(value)) * 1000)


def extract(path):
    with (gzip.open if path.suffix == ".gz" else open)(path, "rt", encoding="utf8") as f:
        data = json.load(f)
    events = data["traceEvents"] if isinstance(data, dict) else data
    hardware = {e["pid"] for e in events if e.get("name") == "process_name"
                and e.get("args", {}).get("name") == "Ascend Hardware"}
    markers = []
    work = []
    for e in events:
        if e.get("ph") != "X":
            continue
        start = ts(e["ts"])
        end = start + ts(e["dur"])
        name = e.get("name", "")
        if name.startswith("PR42634/compiled_control/"):
            markers.append((start, end, name.rsplit("/", 1)[-1]))
        payload = e.get("args", {})
        task = str(payload.get("Task Type", ""))
        if e["pid"] in hardware and task.startswith(("KERNEL_", "AI_", "MIX_", "AIC", "AIV", "SDMA",
                                                     "HDMA", "MEMCPY", "MEMSET")):
            work.append((start, end, str(payload.get("Physic Stream Id")), task, name))
    rows = []
    for start, end, shape in sorted(markers):
        tasks = [dict(stream=s, kind=k, name=n) for t, u, s, k, n in sorted(work) if start <= t and u <= end]
        straddling = sum(1 for t, u, *_ in work if t < end and u > start and not (start <= t and u <= end))
        rows.append(dict(shape=shape, tasks=tasks, boundary_crossing_tasks=straddling))
    return rows


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("a", type=Path)
    cli.add_argument("b", type=Path)
    cli.add_argument("output", type=Path)
    args = cli.parse_args()
    a, b = extract(args.a), extract(args.b)
    same = len(a) == len(b) == 4 and all(x == y and x["tasks"] and not x["boundary_crossing_tasks"]
                                         and all(t["stream"] != "None" for t in x["tasks"])
                                         for x,y in zip(a,b))
    args.output.write_text(json.dumps(dict(association="one compiled call and synchronize per marker; "
                                           "all input generation and output D2H are outside the marker",
                                           same_tasks_and_streams=same, a=a, b=b), indent=2)+"\n")
    return 0 if same else 1


if __name__ == "__main__":
    raise SystemExit(main())
