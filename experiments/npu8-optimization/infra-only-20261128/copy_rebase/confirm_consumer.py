"""Verify every copy-consumer timing window from original metrics and publish the four paired deltas."""
from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from statistics import mean,median,stdev
import sys


def main():
    parser=ArgumentParser()
    parser.add_argument("evidence",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent.parent
    sys.path.insert(0,str(root/"runner"))
    import paired_metrics as pm
    schedule=root/"evidence/archived-runner/dynamic_shapes_npu8.json"
    sh=hashlib.sha256(schedule.read_bytes()).hexdigest()
    ref=json.loads((root/"runner/reference.json").read_bytes())
    assert sh==ref["workspace"]["configs/dynamic_shapes_npu8.json"]
    schedule_rows=pm.read(schedule)
    results=[]
    first_recipe=None
    for pairid in ("copy-consumer-no-profiler-ab","copy-consumer-no-profiler-ba",
                   "copy-consumer-confirm-no-profiler-ba","copy-consumer-confirm-no-profiler-ab"):
        v=pm.paired(args.evidence/"pairs"/pairid,"no-profiler",schedule_rows,args.evidence/"raw",sh)
        if first_recipe is None:first_recipe=v["recipe"]
        assert v["recipe"]==first_recipe
        a,b=v["A"]["ms_per_step"],v["B"]["ms_per_step"]
        results.append(dict(pair_id=pairid,order=v["order"],A=v["A"],B=v["B"],delta_ms=a-b,improvement_pct=100*(1-b/a)))
    deltas=[r["delta_ms"] for r in results]
    report=dict(windows=results,delta_summary_ms=dict(mean=mean(deltas),median=median(deltas),sample_sd=stdev(deltas),min=min(deltas),max=max(deltas)))
    args.out.write_text(json.dumps(report,indent=2)+"\n")

if __name__=="__main__":
    main()
