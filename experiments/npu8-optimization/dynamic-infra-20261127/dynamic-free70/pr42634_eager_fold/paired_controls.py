"""Verify all common optimization counters and stratify aligned Host step deltas by route."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics


FILES=("eager_partition_pack","fast_partition","flat_stats_counts","grouped_side_counts",
       "norm_stats","overflow_scan","side_slab_counts","zero2_local_step","zero_flat_hints",
       "effective_deepspeed_config","model_summary")


def read(path):
    return json.loads(path.read_bytes())


def compare(evidence):
    out={}
    for family in sorted(p for p in evidence.iterdir() if p.is_dir()):
        legs={p.name.split("-")[-1]:p for p in family.glob("[0-9]*") if p.is_dir()}
        a,b=legs["A"],legs["B"]
        env_a,env_b=[read(p/"command.json")["env"] for p in (a,b)]
        row={"environment_differences":{k:[env_a.get(k),env_b.get(k)] for k in sorted(set(env_a)|set(env_b))
                                        if env_a.get(k)!=env_b.get(k)},
             "common_receipts_equal":{n:read(a/"rank000"/(n+".json"))==read(b/"rank000"/(n+".json")) for n in FILES}}
        out[family.name]=row
        if "no-profiler" not in family.name:continue
        metrics=[read(p/"rank000/metrics.json") for p in (a,b)]
        receipt={r["step"]:r for r in read(b/"rank000/eager_fold_plan.json")["steps"]}
        groups=defaultdict(list)
        row["synchronized_steady"]=dict(zip("AB",[m["summary"]["steady_state"] for m in metrics]))
        for x,y in zip(metrics[0]["steps"],metrics[1]["steps"]):
            if x["step"]<metrics[0]["summary"]["steady_state"]["start_step_inclusive"]:continue
            same=("step","batch_size","seq_len")
            if any(x[k]!=y[k] for k in same):
                raise ValueError("unpaired training input")
            groups[receipt[x["step"]]["route"]].append((x["step"],x["host_step_ms"]-y["host_step_ms"]))
        def group(rs):
            vals=[v for _,v in rs]
            return {"n":len(rs),"positive":sum(v>0 for v in vals),"mean_ms":statistics.mean(vals),
                    "median_ms":statistics.median(vals),"min_ms":min(vals),"max_ms":max(vals),"per_step":rs}
        row["paired_host_step_ms_A_minus_B"]={k:group(v) for k,v in groups.items()}
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("evidence",type=Path)
    p.add_argument("out",type=Path)
    args=p.parse_args()
    args.out.write_text(json.dumps(compare(args.evidence),indent=2)+"\n")
