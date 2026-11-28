"""Measure a single eventfd policy change on the verified dynamic embedding recipe."""
import json
from pathlib import Path
import re
import subprocess
import sys

here=Path(__file__).resolve().parent
study=here.parents[1]
name="goal70-queue-on-embedding"
receipt=json.loads((here/"receipt.json").read_bytes())
subprocess.run([sys.executable,str(study/"make_recipe.py"),name,receipt["library"],receipt["sha256"],
                "TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY","--diagnostic-switch","TORCH_NPU_QUEUE_WAKE_AUDIT"],check=True)
for suffix in ("","-audit"):
    p=study/"recipes"/(name+suffix+".json")
    data=json.loads(p.read_bytes())
    data["common"].update(ACLNN_CACHE_LIMIT="65536",NPU_DYNAMIC_REBASE_FORWARD="1",NPU_DYNAMIC_REBASE_INDICES="1")
    p.write_text(json.dumps(data,indent=2))
for mode,order in (("audit","ab"),("no-profiler","ab"),("l0","ab"),("no-profiler","ba"),("l0","ba"),("l1","ab")):
    out=study/"pairs"/f"{name}-{mode}-{order}"
    if not (out/"result.json").exists():
        subprocess.run([sys.executable,"-B","runner/host_pair.py","--runner","archived-runner","--recipe",
                    "recipes/"+name+("-audit" if mode=="audit" else "")+".json","--out",str(out),"--mode",mode,
                    "--order",order],cwd=study,check=True)
    result=json.loads((out/"result.json").read_bytes())
    assert result["exit"]==0
    if mode=="audit":
        assert result["audit_passed"] and result["audited_steps"]==8
        counts=[]
        for leg in ("01-A","02-B"):
            log=(Path(result["directory"])/leg/"run.log").read_text(errors="replace")
            line,=re.findall(r"^PTA_QUEUE_PRODUCER_WAKE device=0 .*",log,re.M)
            counts.append({k:int(v) for k,v in re.findall(r"(\w+)=(\d+)",line)})
        a,b=counts
        assert a["enabled"]==0 and b["enabled"]==1 and b["waiters"]==a["waiters"]==0
        assert a["drained"]==b["drained"] and a["writes_without_observed_waiter"]>0
        if b["registrations"]==0:
            assert b["writes"]==0
        (here/"counts.json").write_text(json.dumps(counts,indent=2))
    print("verified",mode,order,flush=True)
