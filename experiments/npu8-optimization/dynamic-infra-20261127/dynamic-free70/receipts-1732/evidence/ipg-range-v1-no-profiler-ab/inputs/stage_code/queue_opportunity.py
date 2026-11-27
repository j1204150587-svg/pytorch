"""Count adjacent copy requests completely enqueued before the preceding dequeue starts."""
import argparse
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import itertools
import json
from pathlib import Path


def ns(v):
    return int(Decimal(str(v))*1000)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("study",type=Path)
    cli.add_argument("out",type=Path)
    args=cli.parse_args()
    csv_path=args.study/"analysis/cross-copy-l1-ab/marked_copy_calls.csv.gz"
    with gzip.open(csv_path,"rt") as f:
        calls=list(csv.DictReader(f))
    rows=[]
    for leg,sub in (("A","01-A"),("B","02-B")):
        trace=args.study/"evidence/cross-copy-l1-ab"/sub/"trace_view.json.gz"
        with gzip.open(trace,"rt",encoding="utf8") as f: d=json.load(f)
        events=d if isinstance(d,list) else d["traceEvents"]
        en={s["args"]["correlation_id"]:ns(s["ts"])+ns(s["dur"])
            for s in events if s.get("cat")=="enqueue" and s["ph"]=="X"}
        service=defaultdict(list)
        for s in events:
            if s.get("cat")=="dequeue" and s["ph"]=="X":
                service[s["tid"]].append((ns(s["ts"]),s["args"]["correlation_id"]))
        for seq in service.values():seq.sort()
        for step in ("4","5"):
            for cohort in sorted({r["cohort"] for r in calls}):
                ids={int(r["queue_id"]) for r in calls if r["leg"]==leg and r["step"]==step and r["cohort"]==cohort}
                groups=[]
                formed=[]
                for seq in service.values():
                    for flag,chunk in itertools.groupby(seq,key=lambda r:r[1] in ids):
                        if not flag:continue
                        batch=list(chunk); groups.append(len(batch)); i=0
                        while i<len(batch):
                            until=batch[i][0]; j=i+1
                            while j<len(batch) and j-i<64 and en[batch[j][1]]<=until:j+=1
                            formed.append(j-i); i=j
                rows.append(dict(leg=leg,step=int(step),cohort=cohort,total=len(ids),consecutive_groups=len(groups),
                                 longest=max(groups),ready_group_sizes=dict(Counter(formed)),would_save=len(ids)-len(formed)))
    args.out.write_text(json.dumps(rows,indent=2)+"\n")
