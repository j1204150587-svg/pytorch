"""Apply the archived #42/#44 D2D join to the new dynamic baseline Level1."""
import argparse
import importlib.util
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("leg",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--reader",type=Path,default=Path(__file__).resolve().parent/"d2d_reader/analyze_copy.py")
    args=parser.parse_args()
    spec=importlib.util.spec_from_file_location("archive_copy",args.reader)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    mod.RUNS=args.leg.resolve().parent
    mod.OUT=args.output.resolve()
    mod.OUT.mkdir(parents=True,exist_ok=True)
    # This historical join requires ProfilerStep #4 and #5.
    rows,audit=mod.analyze("dynamic_eager",args.leg.resolve().name)
    for r in rows:
        if "AccumulateGrad" in (r["cpu_ancestors"] or ""):
            r["source"]="AccumulateGrad"
        elif r["stage"]=="backward" and r["consecutive_work_tasks"]>=1000:
            r["source"]="backward_contiguous_SDMA_run"
        else:
            r["source"]="other"
    mod.write("d2d_joins.csv",audit)
    mod.write("d2d_calls.csv.gz",rows,compressed=True)
    mod.write("d2d_subgroups.csv",mod.summarize(rows,("step","source")))
    mod.write("d2d_parent.csv",mod.summarize(rows,("step","cpu_ancestors")))


if __name__=="__main__":
    main()
