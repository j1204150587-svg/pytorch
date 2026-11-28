"""Measure original stock versus the already verified cache and two embedding changes."""
import json
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
study=here.parents[1]
recipe=here/"endpoints.json"
name="goal70-stock-to-embedding"
def read(path): return json.loads(path.read_bytes())
rounds={}
for mode,order in (("audit","ab"),("no-profiler","ab"),("l0","ab"),("no-profiler","ba"),("l0","ba"),("l1","ab")):
    out=study/"pairs"/f"{name}-{mode}-{order}"
    if not (out/"result.json").exists():
        subprocess.run([sys.executable,"-B",str(here/"endpoint_pair.py"),"--runner","archived-runner","--recipe",str(recipe),
                        "--out",str(out),"--mode",mode,"--order",order],cwd=study,check=True)
    result=read(out/"result.json")
    assert result["exit"]==0
    if mode=="audit":
        assert result["audit_passed"] and result["audited_steps"]==8
    for leg in sorted(out.glob("0?-?")):
        native=read(recipe)["endpoints"][leg.name[-1]]
        worker=read(leg/"worker.json")
        assert set(worker["libraries"])==set(native["libraries"])
        assert worker["env"][native["switch"]]==native[leg.name[-1]]
        for k,v in native["common"].items():
            assert v in worker["env"][k].split(":") if k=="PYTHONPATH" else v==worker["env"][k]
    rounds[mode,order]=result["directory"]
    print("verified",mode,order,flush=True)
cmd=[sys.executable,"-B",str(study/"runner/paired_metrics.py"),"--schedule",str(study/"archived-runner/dynamic_shapes_npu8.json")]
for order in ("ab","ba"):
    cmd+=["--round",rounds["no-profiler",order],rounds["l0",order]]
with (here/"metrics.json").open("w") as out:
    subprocess.run(cmd,stdout=out,check=True)
