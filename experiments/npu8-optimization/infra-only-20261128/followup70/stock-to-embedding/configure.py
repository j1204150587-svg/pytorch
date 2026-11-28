"""Configure an external host-side two-endpoint check using the archived dynamic runner."""
import hashlib
import json
from pathlib import Path

here=Path(__file__).resolve().parent
study=here.parents[1]
wrapper=study/"runner/host_pair.py"

def read(path):
    return json.loads(path.read_text())

def write(path, obj):
    path.write_text(json.dumps(obj,indent=2)+"\n")

def once(src, a, b):
    assert src.count(a)==1,a
    return src.replace(a,b)

source=wrapper.read_text()
source=once(source,'REFERENCE = json.loads((HERE / "reference.json").read_text())',
                   'REFERENCE = json.loads((HERE.parents[1] / "runner/reference.json").read_text())')
source=once(source,'    recipe = read(receipt_dir / "recipe.json")',
            '    recipe = read(receipt_dir / "recipe.json")["endpoints"][variant]')
source=once(source,'        if "INFRA_COUNTER_DIR" in recipe["common"]:',
            '        if any("INFRA_COUNTER_DIR" in e["common"] for e in recipe["endpoints"].values()):')
# Native guard: record absent stock cache settings in the same live worker map check.
source=once(source,'                namespace = next(line',
'''                if libraries == {"/usr/local/python3.12.13/lib/python3.12/site-packages/torch_npu/lib/libtorch_npu.so":
                                     "6557790a35e6cabf6300e91b8eda3944bc30d8ac834fff48161cf72d0c6e6b39"}:
                    absent = ("ACLNN_CACHE_LIMIT","NPU_DYNAMIC_REBASE_FORWARD","NPU_DYNAMIC_REBASE_INDICES")
                    check(all(key.encode() not in env for key in absent), "unexpected stock override")
                    result["verified_absent"] = list(absent)
                namespace = next(line''')
shim=here/"endpoint_pair.py"
shim.write_text(source)
shim.chmod(0o755)
write(here/"wrapper-hashes.json",{str(wrapper):hashlib.sha256(wrapper.read_bytes()).hexdigest(),
                                  shim.name:hashlib.sha256(shim.read_bytes()).hexdigest()})
write(here/"endpoints.json",dict(endpoints={"A":read(study/"recipes/stock.json"),
                                          "B":read(study/"recipes/emb-fwd.json")}))
