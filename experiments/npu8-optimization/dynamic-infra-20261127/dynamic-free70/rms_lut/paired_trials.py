"""Recompute synchronized windows, activations and per-step physical scopes in the cap and LUT trials."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"pr42634_eager_fold"))
from paired_attribution import leg_report as fold_report
from train_trace import analyze as rms_scopes


def compute(data,mode):
    result={}
    count_file="eager_fold_plan.json" if mode=="cap" else "planned_rms_lut.json"
    for run in sorted(data.iterdir()):
        if not run.is_dir():continue
        legs={}
        for leg in sorted(run.iterdir()):
            proof=leg/"rank000"/count_file
            if not proof.is_file():continue
            counts=json.loads(proof.read_bytes())
            if mode=="rms":
                info={k:counts[k] for k in ("rms","launches","all_expected","expected_forwards","shape_routes")}
            else:
                info={k:counts[k] for k in ("counts","steps")}
            metrics=json.loads((leg/"rank000/metrics.json").read_bytes())
            info["window"]=metrics["summary"].get("steady_state")
            info["host_steps"]=metrics["steps"]
            trace=leg/"trace_view.json.gz"
            if trace.is_file():
                if mode=="rms":
                    info["attributed"]=rms_scopes(trace)["steps"]
                else:
                    info["attributed"]=fold_report(leg)
                with (leg/"step_trace_time.csv").open(encoding="utf-8-sig",newline="") as f:
                    info["official"]=list(csv.DictReader(f))
            legs[leg.name[-1]]=info
        if legs:
            result[run.name]=legs
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("data",type=Path)
    p.add_argument("mode",choices=("rms","cap"))
    p.add_argument("out",type=Path)
    a=p.parse_args()
    a.out.write_text(json.dumps(compute(a.data,a.mode),indent=2)+"\n")
