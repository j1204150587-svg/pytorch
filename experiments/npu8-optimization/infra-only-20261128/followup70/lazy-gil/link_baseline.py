"""Relink the verified dual-embedding wheel with the separately compiled PTA lazy-init object."""
import hashlib
import json
from pathlib import Path
import subprocess


def digest(file):
    with Path(file).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


here = Path(__file__).resolve().parent
parent_path = Path("/artifacts/infra-only-20261128/embedding/build/build-receipt.json")
parent = json.loads(parent_path.read_bytes())
view_path = here / "v1/build-receipt.json"
view = json.loads(view_path.read_bytes())
assert parent["status"] == view["status"] == "built"
expected = "4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"
assert parent["library_sha256"] == digest(parent["library"]) == expected
assert (parent["pta_sha"], parent["op_plugin_sha"]) == (view["pta_sha"], view["op_plugin_sha"])
for obj in parent["compile"] + view["compile"]:
    assert digest(obj["object"]) == obj["object_sha256"]
unit, = view["compile"]
command = list(parent["link"])
cwd = Path(parent["link_cwd"])
index, = [i for i,s in enumerate(command) if s.endswith(".o") and (cwd / s).resolve() == Path(unit["original_object"])]
command[index] = unit["object"]
lib = here / "combined/libtorch_npu.so"
lib.parent.mkdir(exist_ok=False)
command[command.index("-o")+1] = str(lib)
subprocess.run(command,cwd=cwd,check=True)
result = {"parents":[str(parent_path),str(view_path)],"parent_receipts_sha256":[digest(parent_path),digest(view_path)],
          "pta_sha":view["pta_sha"],"op_plugin_sha":view["op_plugin_sha"],"source_blobs":view["original_sources_sha256"],
          "objects":[{"source":u["source"],"object_sha256":u["object_sha256"]} for u in parent["compile"]+view["compile"]],
          "link":command,"cwd":str(cwd),"library":str(lib),"library_sha256":digest(lib),"status":"built"}
(lib.parent/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
print(result["library_sha256"])
