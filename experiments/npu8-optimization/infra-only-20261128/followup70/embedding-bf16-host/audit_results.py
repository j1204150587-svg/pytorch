"""Recompute semantic and physical work summaries from the untouched probe archive."""
import argparse
import array
import collections
import csv
import gzip
import hashlib
import io
import json
import struct
import sys
import tarfile
from decimal import Decimal
from pathlib import Path


def decode(word):
    return struct.unpack(">f", struct.pack(">I", word << 16))[0]


def inspect_outputs(archive, repeat):
    result = []
    with gzip.open(archive.extractfile("repeat/replays.i16.gz"), "rb") as raw:
        for row in repeat["cases"]:
            legs = {"A": [], "B": []}
            for item in row["results"]:
                data = raw.read(8192)
                assert len(data) == 8192
                assert hashlib.sha256(data).hexdigest() == item["output_sha256"]
                vals = array.array("H")
                vals.frombytes(data)
                if sys.byteorder != "little":
                    vals.byteswap()
                legs[item["leg"]].append(vals)
            varying = collections.Counter()
            for idx in range(4096):
                a = {x[idx] for x in legs["A"]}
                b = {x[idx] for x in legs["B"]}
                if len(a|b) > 1:
                    varying["any"] += 1
                    varying["A_variable"] += len(a) > 1
                    varying["B_variable"] += len(b) > 1
                    varying["stable_A_vs_stable_B_different"] += len(a) == len(b) == 1 and a != b
                    varying["observed_sets_disjoint"] += not bool(a & b)
                    varying["observed_values_in_common"] += bool(a & b)
            by_leg = {}
            for leg in "AB":
                stats = row["by_leg"][leg]
                by_leg[leg] = dict(repeats=stats["repetitions"], distinct=stats["distinct_outputs"],
                                   samples=[{str(decode(int(word))): n for word, n in h.items()}
                                            for h in stats["histograms"]])
            result.append(dict(label=row["label"], shape=row["receipt"]["shape"], stream=row["receipt"]["stream"],
                               varying=dict(varying), by_leg=by_leg, points=row["points"]))
        assert raw.read(1) == b""
    return result


def physical(read, names):
    kpath, = [n for n in names if n.startswith("repeat/") and n.endswith("/kernel_details.csv")]
    tpath, = [n for n in names if n.startswith("repeat/") and n.endswith("/trace_view.json")]
    traces = json.loads(read(tpath),parse_float=Decimal)
    if isinstance(traces, dict):
        traces = traces["traceEvents"]
    hw, = [e["pid"] for e in traces if e.get("name")=="process_name" and e.get("args",{}).get("name")=="Ascend Hardware"]
    hardware=[e for e in traces if e.get("ph")=="X" and e["pid"]==hw]
    markers = sorted((e for e in traces if e.get("ph")=="X" and e.get("name","").startswith("EmbeddingFixed_")),
                     key=lambda m: m["ts"])
    with io.StringIO(read(kpath).decode("utf-8-sig")) as text:
        kernels = list(csv.DictReader(text))
    tasks = []
    for m in markers:
        begin, end = Decimal(m["ts"]), Decimal(m["ts"]) + Decimal(m["dur"])
        matches = [r for r in kernels if begin <= Decimal(r["Start Time(us)"].strip()) < end]
        physical = [(r["Type"],r["Name"],r["Stream ID"],r["Block Num"],r["Input Data Types"],r["Output Data Types"])
                    for r in matches]
        actual=[(e["name"],e.get("args",{}).get("Task Type"),e.get("args",{}).get("Physic Stream Id"),
                 e.get("args",{}).get("Task Id")) for e in hardware if begin<=Decimal(e["ts"])<end]
        tasks.append(dict(marker=m["name"], physical=physical,all_hardware=actual))
    return dict(markers=len(markers), actual=tasks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive",type=Path)
    parser.add_argument("output",type=Path)
    args = parser.parse_args()
    with tarfile.open(args.archive,"r:gz") as arc:
        names = arc.getnames()
        def read(name):
            return arc.extractfile(name).read()
        controlled = json.loads(read("controlled/results.json"))
        initial = json.loads(read("probe/results.json"))
        repeat = json.loads(read("repeat/results.json"))
        assert repeat["complete"] and len(controlled["checks"])==22
        assert all(read(p+"/train.exit").strip()==b"0" for p in ("probe","controlled","repeat"))
        workload = repeat["workload_fan_in"]
        agg = collections.Counter()
        for row in workload["steps"]:
            agg.update({int(k):v for k,v in row["tables_by_max_fan_in"].items()})
        analysis = dict(first_pass=sum(not any(r["different_elements_AA_BB_AB_AB"]) for r in initial["rows"]),
                        initial_tail=initial["rows"][-1], controlled_passed=controlled["passed"],
                        controlled=controlled["checks"], fixed=inspect_outputs(arc,repeat),
                        physical=physical(read,names), tables_total=sum(agg.values()),
                        tables_max_fan_in_hist=dict(sorted(agg.items())), tables_at_most_two=sum(v for k,v in agg.items() if k<=2))
        args.output.write_text(json.dumps(analysis,indent=2),encoding="utf-8")
        print("controlled, packet, profile labels, tables <=2",controlled["passed"],len(repeat["cases"]),
              analysis["physical"]["markers"],analysis["tables_at_most_two"],"/",sum(agg.values()))


if __name__=="__main__":
    main()

