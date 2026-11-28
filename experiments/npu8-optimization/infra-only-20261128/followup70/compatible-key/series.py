"""Require a matching option lifetime and two-sided micro gain before training timings."""
import json
from pathlib import Path
import re
import subprocess
import sys
import time

here = Path(__file__).resolve().parent
study = here.parents[1]
name = "goal70-compatible-key"
def read(path):
    return json.loads(path.read_bytes())

deadline = time.monotonic()+300
while not (here/"cpu.json").exists():
    if time.monotonic() > deadline: raise TimeoutError("CPU gate receipt")
    time.sleep(2)
gate = read(here/"cpu.json")
assert [r["key"] for r in gate] == list("0110")
if not (gate[1]["median_ns"] < gate[0]["median_ns"] and gate[2]["median_ns"] < gate[3]["median_ns"]):
    print("CPU micro did not improve in both orders",flush=True)
    sys.exit(0)
receipt = read(here/"build/build-receipt.json")
switch = "TORCH_NPU_STABLE_COMPATIBLE_KEY"
subprocess.run([sys.executable,str(study/"make_recipe.py"),name,receipt["library"],receipt["sha256"],switch,
                "--diagnostic-switch","TORCH_NPU_COMPATIBLE_QUERY_AUDIT"],check=True)
for suffix in ("","-audit"):
    file = study/"recipes"/(name+suffix+".json")
    recipe = read(file)
    recipe["common"].update(ACLNN_CACHE_LIMIT="65536",NPU_DYNAMIC_REBASE_FORWARD="1",NPU_DYNAMIC_REBASE_INDICES="1")
    file.write_text(json.dumps(recipe,indent=2))
for mode,order in (("audit","ab"),("no-profiler","ab"),("l0","ab"),("no-profiler","ba"),("l0","ba"),("l1","ab")):
    out = study/"pairs"/f"{name}-{mode}-{order}"
    if not (out/"result.json").exists():
        recipe = name + ("-audit" if mode=="audit" else "")
        subprocess.run([sys.executable,"-B","runner/host_pair.py","--runner","archived-runner","--recipe",
                        "recipes/"+recipe+".json","--out",str(out),"--mode",mode,"--order",order],cwd=study,check=True)
    result = read(out/"result.json")
    assert result["exit"] == 0
    if mode=="audit":
        assert result["audit_passed"] and result["audited_steps"]==8
        audit = {}
        for leg in ("01-A","02-B"):
            log = (Path(result["directory"])/leg/"run.log").read_text(errors="replace")
            rows = [[int(x) for x in r] for r in re.findall(r"NPU_COMPATIBLE_QUERY calls=(\d+) ns=(\d+)",log)]
            assert rows and sum(r[0] for r in rows)>0
            audit[leg] = dict(threads=rows, queries=sum(r[0] for r in rows), ns=sum(r[1] for r in rows))
        (here/"counts.json").write_text(json.dumps(audit,indent=2))
    print(mode, order, result["directory"],flush=True)
