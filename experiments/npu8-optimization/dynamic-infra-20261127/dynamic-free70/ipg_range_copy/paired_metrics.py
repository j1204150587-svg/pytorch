"""Recompute pooled steady-state and profiler windows for a fully collected family."""
import argparse
import csv
import json
from pathlib import Path


def measures(d):
    m = json.loads((d/"metrics.json").read_text())
    if m["summary"].get("steady_state"):
        s = m["summary"]["steady_state"]
        assert (s["steps"],s["samples"],s["tokens"]) == (50,136,2994)
        return {k:s[k] for k in ("steps","samples","tokens","elapsed_s_synchronized","ms_per_step","samples_per_s","tokens_per_s")}
    with (d/"step_trace_time.csv").open() as f:
        r = [v for v in csv.DictReader(f) if v["Step"] in ("4","5","6")]
    assert [v["Step"] for v in r] == ["4","5","6"]
    shapes = [[m["steps"][i][k] for k in ("batch_size","seq_len")] for i in (4,5,6)]
    assert shapes == [[1,19],[2,14],[2,22]]
    t = {k:[float(v[k])/1000 for v in r] for k in ("Stage","Free","Computing")}
    return dict(shapes=shapes,steps=3,per_step_ms=t,**{k:sum(v)/3 for k,v in t.items()},
                free_share=sum(t["Free"])/sum(t["Stage"]))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",type=Path)
    args=p.parse_args()
    d=args.folder
    runs=json.loads((d/"runs.json").read_text())
    pairs={"no-profiler":[],"l0":[]}
    avg={}
    for run in runs:
        if run["mode"] not in pairs or run["exit"] != 0:
            continue
        pair={path.name[-1]:measures(path) for path in sorted((d/run["label"]).glob("??-?"))}
        a,b=pair["A"],pair["B"]
        key = "ms_per_step" if run["mode"]=="no-profiler" else "free_share"
        pair["change_pct"]=(a[key]/b[key]-1)*100 if key=="ms_per_step" else (1-b[key]/a[key])*100
        pair["order"]=run["order"]
        pairs[run["mode"]].append(pair)
    for typ,collection in pairs.items():
        assert sorted(x["order"] for x in collection)==["ab","ba"]
        keys=["ms_per_step"] if typ=="no-profiler" else ["Free","Stage","Computing"]
        means={leg:{k:sum(x[leg][k] for x in collection)/2 for k in keys} for leg in ("A","B")}
        for row in means.values():
            if typ=="no-profiler":
                row.update(samples_s=136*1000/(50*row["ms_per_step"]),tokens_s=2994*1000/(50*row["ms_per_step"]))
            else:
                row["free_share"]=row["Free"]/row["Stage"]
        avg[typ]=means
    avg["throughput_change_pct"]=(avg["no-profiler"]["A"]["ms_per_step"]/avg["no-profiler"]["B"]["ms_per_step"]-1)*100
    avg["free_share_change_pct"]=(1-avg["l0"]["B"]["free_share"]/avg["l0"]["A"]["free_share"])*100
    res=dict(averages=avg,pairs=pairs)
    (d/"recomputed.json").write_text(json.dumps(res,indent=2)+"\n")
    print(json.dumps(avg))


if __name__=="__main__":
    main()
