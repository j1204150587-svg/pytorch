"""Check hardware slot identity and describe HAL service time by the preceding arrival gap."""
from collections import Counter,defaultdict
import csv
import gzip
import json
from pathlib import Path
import statistics

root=Path(__file__).parent
with gzip.open(root/"calls.csv.gz","rt",encoding="utf-8-sig",newline="") as fp:
    rows=list(csv.DictReader(fp))
groups=defaultdict(list)
for r in rows:
    groups[r["step"],r["sq_id"]].append(r)
bands=(2,5,10,20,50,100)
report={}
for (step,sq),group in sorted(groups.items()):
    group.sort(key=lambda r:int(r["start_ns"]))
    offsets=Counter((int(r["task_id"])-int(r["driver_position"]))%2048 for r in group if r["task_id"])
    values=defaultdict(list)
    for prev,nxt in zip(group,group[1:]):
        dt=(int(nxt["start_ns"])-int(prev["end_ns"]))/1000
        assert dt>=0
        bound=next((f"below_{n}_us" for n in bands if dt<n),"100_us_or_more")
        values[bound].append(int(nxt["ns"])/1000)
    report[f"step_{step}_SQ_{sq}"]=dict(observed_task_minus_position_mod2048=dict(offsets),
        gap_since_last_HAL_return={k:dict(calls=len(v),median_us=statistics.median(v)) for k,v in values.items()})
(root/"arrival.json").write_text(json.dumps(report,indent=2))
