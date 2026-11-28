"""Submit a dedicated Level1 observer pair to the existing two NPU8 locks."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
here=Path(__file__).resolve().parent
study=here.parents[1]
inn="/artifacts/"+str(here.relative_to(study.parent))
(study/"recipes").mkdir(exist_ok=True)
(here/"capture-v2").mkdir(exist_ok=True)
recipe=json.loads((study/"recipes/emb-fwd.json").read_text())
recipe["common"].update(NPU_DYNAMIC_REBASE_FORWARD="1",NPU_HAL_SUBMIT_LOG=inn+"/capture-v2",LD_PRELOAD=inn+"/probe-v2.so")
recipe["switch"]="NPU_HAL_SUBMIT_OBSERVE"
recipe["libraries"][inn+"/probe-v2.so"]=hashlib.sha256((here/"probe-v2.so").read_bytes()).hexdigest()
out=study/"pairs/goal70-hal-submit-v2-l1-ab"
target=study/"recipes/goal70-hal-submit-v2.json"
target.write_text(json.dumps(recipe,indent=2))
if not (out/"result.json").exists():
    subprocess.run([sys.executable,"-B","runner/host_pair.py","--runner","archived-runner","--recipe",str(target),
                   "--out",str(out),"--mode","l1","--order","ab"],cwd=study,check=True)
result=json.loads((out/"result.json").read_bytes())
assert result["exit"]==0
print(result["directory"])
