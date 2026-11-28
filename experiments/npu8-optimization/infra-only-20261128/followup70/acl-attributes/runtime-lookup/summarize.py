"""Summarize per-round paired mean time per call and multiply by measured HSTU call counts."""
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

root=Path(__file__).resolve().parent
counts=(9892,9893)
summary={}
for fname in sorted(root.glob("cpu-*.csv")):
    rows=list(csv.DictReader(fname.open()))
    groups=defaultdict(list)
    for r in rows:
        assert int(r["cpu"])==int(fname.stem.split("-")[-1])
        groups[int(r["batch"]),int(r["round"])].append(r)
    results={}
    for batch in sorted({b for b,_ in groups}):
        durations=defaultdict(list)
        for rnd in range(51):
            block=groups[batch,rnd]
            assert "".join(r["leg"] for r in block)==("ABBA" if rnd%2==0 else "BAAB")
            for clock in ("wall_ns","cpu_ns"):
                a,b=(mean(int(r[clock])/batch for r in block if r["leg"]==leg) for leg in "AB")
                for leg,t in (("A",a),("B",b),("A_minus_B",a-b)):
                    durations[clock,leg].append(t)
        desc={}
        for (clock,leg),samples in durations.items():
            s=sorted(samples)
            desc[f"{clock}_{leg}"]={"rounds":len(s),"p05":s[2],"p50":median(s),"p95":s[47]}
        m=desc["wall_ns_A_minus_B"]["p50"]
        desc["projected_step_ms"]=[m*c/1e6 for c in counts]
        results[batch]=desc
    summary[fname.name]=results
(root/"timing.json").write_text(json.dumps(summary,indent=2)+"\n")
for core,t in summary.items():
    for batch,result in t.items():
        print(core,batch,result["wall_ns_A"]["p50"],result["wall_ns_B"]["p50"],result["wall_ns_A_minus_B"], result["projected_step_ms"])
