"""Record exactly one changing descriptor policy per pair, both with the larger CANN cache."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
digest = sys.argv[1]
library = "/artifacts/infra-only-20261128/embedding/build/package/torch_npu/lib/libtorch_npu.so"
subprocess.run([sys.executable, str(ROOT/"make_recipe.py"), "embedding", library, digest,
                "NPU_DYNAMIC_REBASE_INDICES"], check=True)
probe = ROOT/"counter_embedding.so"
probe_path = "/artifacts/infra-only-20261128/" + probe.name
for name, key, prerequisite in (("emb-back", "NPU_DYNAMIC_REBASE_INDICES", {"NPU_DYNAMIC_REBASE_FORWARD":"0"}),
                                 ("emb-fwd", "NPU_DYNAMIC_REBASE_FORWARD", {"NPU_DYNAMIC_REBASE_INDICES":"1"})):
    for audit in (False, True):
        cfg = json.loads((ROOT/"recipes/embedding.json").read_bytes())
        cfg["switch"] = key
        cfg["common"].update(prerequisite, ACLNN_CACHE_LIMIT="65536", NPU_DYNAMIC_REBASE_COUNT=str(int(audit)))
        if audit:
            cfg["common"].update(LD_PRELOAD=probe_path, INFRA_COUNTER_DIR="{output}")
            cfg["libraries"][probe_path]=hashlib.sha256(probe.read_bytes()).hexdigest()
        (ROOT/"recipes"/(name+("-audit" if audit else "")+".json")).write_text(json.dumps(cfg, indent=2)+"\n")

