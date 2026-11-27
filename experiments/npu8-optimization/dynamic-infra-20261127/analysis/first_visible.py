"""Count which observable boundary of the next request was pending at gap onset."""
import argparse
from collections import defaultdict
import csv
import gzip
from pathlib import Path
from analyze_dynamic_trace import stats,write_csv


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis",type=Path)
    args=parser.parse_args()
    groups=defaultdict(list)
    with gzip.open(args.analysis/"device_gaps.csv.gz","rt",newline="") as f:
        for row in csv.DictReader(f):
            if row["state"]!="linked":
                continue
            field=row["first_visible"]
            for op,role in (("ALL","ALL"),(row["op"],row["role"])):
                groups[row["step"],op,role,field].append((int(row["gap_ns"]),int(row[field])))
    out=[]
    for (step,op,role,field),values in sorted(groups.items()):
        out.append(dict(step=step,op=op,role=role,first_visible=field,
                        onset_component_ms=sum(v for _,v in values)/1e6,**stats([g for g,_ in values])))
    write_csv(args.analysis/"first_visible.csv",out)


if __name__=="__main__":
    main()
