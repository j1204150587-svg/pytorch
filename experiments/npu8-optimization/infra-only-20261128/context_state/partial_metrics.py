"""Recompute the completed context-state matrix, including the unprofiled reverse pair."""
import importlib.util
import json
from pathlib import Path
from statistics import mean, stdev

root=Path(__file__).resolve().parent
suite=root.parent
spec=importlib.util.spec_from_file_location("paired_metrics",suite/"runner/paired_metrics.py")
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
schedule=suite/"evidence/archived-runner/dynamic_shapes_npu8.json"
digest=m.hashlib.sha256(schedule.read_bytes()).hexdigest()
assert digest==m.read(suite/"runner/reference.json")["workspace"]["configs/dynamic_shapes_npu8.json"]
shapes=m.read(schedule)
out=root/"analysis"
out.mkdir(exist_ok=True)
r={}
for kind,order in (("no-profiler","ab"),("no-profiler","ba"),("l0","ab")):
    pair=root/"evidence/pairs"/f"context-state-{kind}-{order}"
    value=m.paired(pair,kind,shapes,root/"evidence/raw",digest)
    assert value["order"]==order
    r[f"{kind}-{order}"]=value
assert len({json.dumps(v["recipe"],sort_keys=True) for v in r.values()})==1
deltas=[]
for order in ("ab","ba"):
    t=r["no-profiler-"+order]
    a,b=[t[v]["ms_per_step"] for v in "AB"]
    t["delta_ms_per_step"]=a-b
    t["gain_pct"]=m.gain(a,b)
    deltas.append(a-b)
l0=r["l0-ab"]
l0["free_ratio_gain_pct"]=m.gain(l0["A"]["free_ratio"],l0["B"]["free_ratio"])
r["paired_delta_ms_per_step"]={"mean":mean(deltas),"sample_sd":stdev(deltas),"rounds":2}
r["l0-ba"]={"status":"cancelled_while_waiting_for_admission","measured_legs":0}
(out/"paired.json").write_text(json.dumps(r,indent=2)+"\n")
for k in ("no-profiler-ab","no-profiler-ba","l0-ab","paired_delta_ms_per_step"):
    print(k,{name:detail for name,detail in r[k].items() if name in ("A","B","delta_ms_per_step","gain_pct","free_ratio_gain_pct","mean","sample_sd")})
