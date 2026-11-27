"""Recompute two dynamic Level1 legs and compare ordered submissions and latency."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from analyze_dynamic_trace import write_csv


def load_rows(path, key):
    with path.open(encoding="utf-8", newline="") as handle:
        return {tuple(row[k] for k in key):row for row in csv.DictReader(handle)}


def compare_rows(out, filename, keys, steps):
    a, b = (load_rows(out / leg / filename,keys) for leg in ("A","B"))
    rows = []
    for key in sorted(a.keys()|b.keys()):
        if steps and int(key[0]) not in steps:
            continue
        x,y = a.get(key,{}),b.get(key,{})
        row = {k:v for k,v in zip(keys,key)}
        row.update(batch=(x or y)["batch"],seq_len=(x or y)["seq_len"])
        for field in ("n","sum_ms","p50_us","p95_us","p99_us","max_us"):
            row["A_"+field]=x.get(field,"")
            row["B_"+field]=y.get(field,"")
            if x.get(field) and y.get(field):
                row["B_minus_A_"+field] = round(float(y[field])-float(x[field]),6)
        rows.append(row)
    write_csv(out/filename,rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("a",type=Path,help="baseline leg including rank000/metrics.json")
    parser.add_argument("b",type=Path,help="candidate leg")
    parser.add_argument("output",type=Path)
    parser.add_argument("--steps",nargs="+",type=int,help="optional latency CSV step selection; all profiled steps by default")
    args = parser.parse_args()
    here=Path(__file__).resolve().parent
    args.output.mkdir(parents=True,exist_ok=True)
    for name,leg in (("A",args.a),("B",args.b)):
        dest=args.output/name
        subprocess.run([sys.executable,"-B",str(here/"analyze_dynamic_trace.py"),str(leg),str(dest)],check=True)
    report_path=args.output/"submissions_compare.json"
    subprocess.run([sys.executable,"-B",str(here/"compare_submissions.py"),
                    str(args.output/"A/submissions.json"),str(args.output/"B/submissions.json"),str(report_path)],check=True)
    available = {s["step"] for s in json.loads(report_path.read_bytes())["steps"]}
    if args.steps and not set(args.steps).issubset(available):
        raise ValueError(f"requested {args.steps} but trace paired {sorted(available)}")
    compare_rows(args.output,"operator_latency.csv",("step","op","aten","role","metric"),args.steps)
    compare_rows(args.output,"gap_attribution.csv",("step","op","role","metric"),args.steps)


if __name__=="__main__":
    main()
