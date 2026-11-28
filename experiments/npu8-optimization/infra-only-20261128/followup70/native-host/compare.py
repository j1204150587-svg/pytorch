"""Produce candidate-specific measurements against identical dynamic eager work."""
import argparse
from collections import Counter,defaultdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence",type=Path,help="directory with extracted pairs/ and raw/")
    parser.add_argument("prefix")
    parser.add_argument("output",type=Path)
    args=parser.parse_args()
    here=Path(__file__).resolve().parent
    infra=here.parents[1]
    study=infra.parent
    spec=importlib.util.spec_from_file_location("producer_exposure",here/"exposure.py")
    producer=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(producer)
    sys.path.insert(0,str(infra/"queue"))
    from analyze_pair import compare, legdata, readcsv
    analyzer=study/"dynamic-infra-20261127/analysis/analyze_dynamic_trace.py"
    pair=args.evidence/"pairs"/(args.prefix+"-l1-ab")
    run=args.evidence/"raw"/Path(json.loads((pair/"result.json").read_bytes())["directory"]).name
    legs={p.name[-1]:p for p in run.glob("0?-?")}
    args.output.mkdir(parents=True,exist_ok=True)
    summary={}
    raw={}
    slots={}
    for key,leg in legs.items():
        dest=args.output/key
        raw[key]=legdata(leg,dest,analyzer)
        slots[key]=defaultdict(dict)
        for row in readcsv(dest/"linked","work_tasks"):
            slots[key][int(row["step"])][row["stream"],int(row["stream_work_slot"])]=(row["task_type"],row["task"])
        cpu=producer.analyze(leg/"trace_view.json.gz",dest/"linked")
        steps={}
        for n,row in raw[key]["steps"].items():
            exposed=cpu[n]["linked_exposure"]
            steps[n]={"shape":row["shape"],"first_seen":row["first_seen"],"all_submissions":row["all"],
                      "physical":{sid:{"work_n":detail["work_n"],"work_sha256":detail["work_sha256"]}
                                  for sid,detail in row["physical"].items()},"acl":row["acl_api"],
                      "launch_us":row["exposure"]["single_acl_launch_us"],"ops":{}}
            for op in ("aten::view","empty_tensor","aten::empty_strided"):
                count=cpu[n]["counts"].get(op,0)
                steps[n]["ops"][op]={"calls":count,"latency_us":cpu[n]["latency_us"].get(op),
                                     "exclusive_cpu_ms":cpu[n]["exclusive_cpu_by_leaf"].get(op,0)/1e6,
                                     "before_aten_gap_ms":exposed["before_next_aten"]["leaf_ns"].get(op,0)/1e6,
                                     "before_enqueue_gap_ms":exposed["before_next_enqueue"]["leaf_ns"].get(op,0)/1e6}
        summary[key]={"trace_sha256":raw[key]["trace_sha256"],"steps":steps}
    summary["compare"]={s:compare(row,raw["B"]["steps"][s]) for s,row in raw["A"]["steps"].items()}
    for step,checks in summary["compare"].items():
        a,b=slots["A"][step],slots["B"][step]
        pairs=a.keys() & b.keys()
        subs=Counter((a[k],b[k]) for k in pairs if a[k]!=b[k])
        checks["physical_slots"]={"A":len(a),"B":len(b),"paired":len(pairs),"identical":len(pairs)-sum(subs.values()),
                                 "substitutions":[{"A":before,"B":after,"count":n} for (before,after),n in subs.most_common()]}
    (args.output/"level1.json").write_text(json.dumps(summary,indent=2)+"\n")


if __name__=="__main__":
    main()
