from collections import Counter
import csv
import json
from pathlib import Path
import statistics


here=Path(__file__).resolve().parent
names=json.loads((here/"request-names.json").read_bytes())
req={s:dict(total=len(seq),aclnn=sum(n.startswith("aclnn") for n in seq),
            other=Counter(n for n in seq if not n.startswith("aclnn"))) for s,seq in names.items()}
cpu=list(csv.DictReader((here/"cpu.csv").open()))
micro={}
for cap in ("64","8192"):
    for heap in ("0","1"):
        rows=[x for x in cpu if x["slots"]==cap and x["heap_capture"]==heap]
        assert len(rows)==64
        med={arm:statistics.median(float(x["ns_per_release_handoff"]) for x in rows if x["arm"]==arm) for arm in "AB"}
        blocks=[]
        for i in range(0,len(rows),4):
            batch=rows[i:i+4]
            assert "".join(r["arm"] for r in batch)=="ABBA"
            arms=[statistics.mean(float(r["ns_per_release_handoff"]) for r in batch if r["arm"]==arm) for arm in "AB"]
            blocks.append(arms[0]-arms[1])
        gain=med["A"]-med["B"]
        micro[f"{cap}-heap{heap}"]=dict(median_ns=med,A_minus_B_ns=gain,ABBA_deltas_ns=blocks,
             positive_ABBA_blocks=sum(v>0 for v in blocks),ABBA_sample_sd=statistics.stdev(blocks),
             budget_ms_at_L1_aclnn_name_counts={s:z["aclnn"]*gain/1e6 for s,z in req.items()})
(here/"cpu-summary.json").write_text(json.dumps(dict(shapes={"4":[1,19],"5":[2,14]},request_counts=req,
    stride_bytes=int(cpu[0]["stride_bytes"]),payload_bytes=int(cpu[0]["payload_bytes"]),micro=micro),indent=2)+"\n")
