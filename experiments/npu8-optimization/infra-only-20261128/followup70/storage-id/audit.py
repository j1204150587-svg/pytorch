"""Eight-step stock HSTU gate after the isolated constructor uniqueness checks."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

here=Path(__file__).resolve().parent
root=here.parents[1]
rows=json.loads((here/"cpu.json").read_bytes())
assert len(rows)==15 and all(r["nonconsecutive"]==0 for r in rows)
link=json.loads((here/"combined/receipt.json").read_bytes())
name="storage-id"
subprocess.run([sys.executable,"-B","make_recipe.py",name,link["library"],link["library_sha256"],
                "TORCH_NPU_STORAGE_ATOMIC_ID","--diagnostic-switch","TORCH_NPU_STORAGE_ID_AUDIT"],cwd=root,check=True)
for audit in (False,True):
    dest=root/"recipes"/(name+("-audit" if audit else "")+".json")
    recipe=json.loads(dest.read_bytes())
    recipe["common"].update(ACLNN_CACHE_LIMIT="65536",NPU_DYNAMIC_REBASE_INDICES="1",NPU_DYNAMIC_REBASE_FORWARD="1",
                            NPU_DYNAMIC_REBASE_COUNT=str(int(audit)))
    if audit:
        probe="/artifacts/infra-only-20261128/counter_embedding.so"
        recipe["common"].update(LD_PRELOAD=probe,INFRA_COUNTER_DIR="{output}")
        recipe["libraries"][probe]=hashlib.sha256((root/"counter_embedding.so").read_bytes()).hexdigest()
    dest.write_text(json.dumps(recipe,indent=2)+"\n")
pair=root/"pairs"/(name+"-audit-ab")
if not (pair/"result.json").exists():
    subprocess.run([sys.executable,"-B","runner/host_pair.py","--runner","archived-runner",
                    "--recipe","recipes/storage-id-audit.json","--out",pair,"--mode","audit","--order","ab"],cwd=root,check=True)
entry=json.loads((pair/"result.json").read_bytes())
assert entry["audit_passed"] and entry["audited_steps"]==8 and entry["exit"]==0
counts={}
for leg in ("01-A","02-B"):
    raw=Path(entry["directory"])/leg
    match,=re.findall(r"PTA_STORAGE_ID enabled=(\d+) calls=(\d+) elapsed_ns=(\d+) max_ns=(\d+)",
                      (raw/"run.log").read_text(errors="replace"))
    enabled,n,elapsed,longest=map(int,match)
    assert enabled==(leg[-1]=="B") and n>50000
    for direction in ("forward","backward"):
        assert entry["native_counters"][leg[-1]]["npu_embedding_"+direction+"_rebase_counter"]["accepted"]==8192
    counts[leg]=dict(enabled=enabled,constructors=n,elapsed_ns=elapsed,max_ns=longest)
(here/"audit.json").write_text(json.dumps(dict(constructors=counts,gate=entry),indent=2)+"\n")
print(counts,flush=True)
