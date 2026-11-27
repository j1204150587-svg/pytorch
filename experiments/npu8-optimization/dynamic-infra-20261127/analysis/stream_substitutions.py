"""Compare each physical stream slot both ways, preserving every observed name substitution."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path


def read(path):
    rows={}
    with gzip.open(path,"rt",newline="") as f:
        for r in csv.DictReader(f):
            key=int(r["step"]),int(r["stream"]),int(r["stream_work_slot"])
            if key in rows:
                raise ValueError(f"duplicate physical slot {key}")
            rows[key]=(r["task_type"],r["task"])
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair",type=Path)
    args=parser.parse_args()
    a,b=(read(args.pair/leg/"work_tasks.csv.gz") for leg in ("A","B"))
    rows=[]
    for step,stream in sorted({k[:2] for k in a.keys()|b.keys()}):
        aa={k[2]:v for k,v in a.items() if k[:2]==(step,stream)}
        bb={k[2]:v for k,v in b.items() if k[:2]==(step,stream)}
        common=aa.keys()&bb.keys()
        pairs=Counter((aa[k],bb[k]) for k in common if aa[k]!=bb[k])
        examples=sorted(k for k in common if aa[k]!=bb[k])[:8]
        rows.append(dict(step=step,stream=stream,a_count=len(aa),b_count=len(bb),
            missing_A_slots=sorted(bb.keys()-aa.keys()),missing_B_slots=sorted(aa.keys()-bb.keys()),
            identical_slots=sum(aa[k]==bb[k] for k in common),substitutions=[
                dict(a=list(x),b=list(y),n=n) for (x,y),n in sorted(pairs.items())],
            first_changed_slots=examples,reverse_consistent=Counter((bb[k],aa[k]) for k in common if aa[k]!=bb[k])==
                Counter({(y,x):n for (x,y),n in pairs.items()})))
    (args.pair/"physical_stream_substitutions.json").write_text(json.dumps(rows,indent=2))


if __name__=="__main__":
    main()
