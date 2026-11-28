"""Admit the storage-run variant using real bitwise, same-source and original-training gates."""
import json
from pathlib import Path
import re
import subprocess
import sys

here = Path(__file__).resolve().parent
study = here.parents[1]
inner = "/artifacts/"+study.name
name = "goal70-flatten-runs"
def read(p):
    return json.loads(p.read_bytes())
receipt = read(here/"build/receipt.json")
subprocess.run([sys.executable,str(study/"make_recipe.py"),name,receipt["library"],receipt["sha256"],
    "TORCH_NPU_FLATTEN_STORAGE_RUNS","--diagnostic-switch","TORCH_NPU_FLATTEN_RUNS_AUDIT"],check=True)
for suffix in ("","-audit"):
    p = study/"recipes"/(name+suffix+".json")
    recipe = read(p)
    recipe["common"].update(ACLNN_CACHE_LIMIT="65536",NPU_DYNAMIC_REBASE_FORWARD="1",NPU_DYNAMIC_REBASE_INDICES="1")
    p.write_text(json.dumps(recipe,indent=2))
micro = []
for i,arm in enumerate("ABBA"):
    dest = here/f"probe-{i+1}-{arm}"
    recipe = read(study/"recipes"/(name+".json"))
    package = recipe["common"]["PYTHONPATH"]
    cmd = ["flock","-x","/run/lock/recsys-gr-ds019-npu8-infra-study.lock",sys.executable,"-B",
           str(study/"operator_probe.py"),"--package",package]
    cmd += [str(dest),inner+"/followup70-20261128/flatten-runs/micro.py","@OUT/result.json",
            "@WORK/configs/dynamic_shapes_npu8.json",inner+"/recipes/"+name+".json",arm]
    if not (dest/"train.exit").exists():
        subprocess.run(cmd,check=True)
    assert (dest/"train.exit").read_text().strip()=="0"
    micro.append(read(dest/"result.json"))
assert len({v["data"] for v in micro})==1 and all(v["cases"]==240 and v["grad_equal"] for v in micro)
if micro[1]["median_ms"]>=micro[0]["median_ms"] and micro[2]["median_ms"]>=micro[3]["median_ms"]:
    print("Both isolated orders regress; stopped at micro gate.",flush=True)
    sys.exit(0)
for mode,order in (("audit","ab"),("l1","ab"),("no-profiler","ab"),("l0","ab"),("no-profiler","ba"),("l0","ba")):
    out = study/"pairs"/f"{name}-{mode}-{order}"
    if not (out/"result.json").exists():
        subprocess.run([sys.executable,"-B","runner/host_pair.py","--runner","archived-runner","--recipe",
            "recipes/"+name+("-audit" if mode=="audit" else "")+".json","--out",str(out),"--mode",mode,"--order",order],
            cwd=study,check=True)
    result = read(out/"result.json")
    assert result["exit"]==0
    if mode=="audit":
        assert result["audit_passed"] and result["audited_steps"]==8
        counts={}
        for leg in ("01-A","02-B"):
            lines = (Path(result["directory"])/leg/"run.log").read_text(errors="replace").splitlines()
            counts[leg] = [{k:int(v) for k,v in re.findall(r"(\w+)=(\d+)",line)} for line in lines if "PTA_FLATTEN_RUNS" in line]
        (here/"counts.json").write_text(json.dumps(counts,indent=2))
        if sum(row["active"] for row in counts["02-B"] if row["inputs"]==1052)<8:
            print("Original 1052-input optimizer did not activate in all eight updates.",flush=True)
            break
    print("verified",mode,order,flush=True)
