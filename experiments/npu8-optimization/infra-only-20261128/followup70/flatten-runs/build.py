"""Build the tested incremental PTA patch on the exact bidirectional embedding objects."""
import difflib
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
study = here.parents[1]
parent = json.loads((study / "embedding/build/build-receipt.json").read_bytes())
stage = here / "build"
stage.mkdir(exist_ok=True)
spec = importlib.util.spec_from_file_location("builder", study.parent / "dynamic-infra-20261127/build_native.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
sha = builder.digest
assert sha(Path(parent["library"])) == parent["library_sha256"] == "4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"
for obj in parent["compile"]:
    assert sha(Path(obj["object"])) == obj["object_sha256"]
checkout = Path(parent["base"]) / "checkout"
db = json.loads((checkout/"build/compile_commands.json").read_bytes())
target = "torch_npu/csrc/aten/ops/FlattenDenseTensorsKernelNpu.cpp"
original = (checkout/target).read_bytes()
assert hashlib.sha1(b"blob "+str(len(original)).encode()+b"\0"+original).hexdigest() == "068d6230cc79330be44fc2e1cbf38d8f4f6b542b"
def apply(src):
    anchor = "at::Tensor NPUNativeFunctions::flatten_dense_tensors"
    marker = "    if (tensors.size() == 1) {"
    assert src.count(anchor) == src.count(marker) == 1
    return ('#include <ATen/TensorSubclassLikeUtils.h>\n#include <cstdio>\n#include <cstdlib>\n#include <cstring>\n'
            + src.replace(anchor,(here/"runs.inc").read_text()+"\n"+anchor)
                 .replace(marker,"    auto combined = tryRunFlatten(tensors);\n    if (combined.defined()) {\n        return combined;\n    }\n"+marker))
def unified(src):
    return "".join(difflib.unified_diff(src.splitlines(True),apply(src).splitlines(True),fromfile="a/"+target,tofile="b/"+target))
dst = stage/target
dst.parent.mkdir(parents=True,exist_ok=True)
dst.write_text(apply(original.decode()))
patch = here/"flatten-runs.patch"
patch.write_text(unified(original.decode()))
stock = subprocess.check_output(["git","-c","safe.directory="+str(checkout),"-C",str(checkout),"show",parent["pta_sha"]+":"+target]).decode()
(here/"from-stock.patch").write_text(unified(stock))
unit = builder.compile_unit(db,checkout,stage,target)
subprocess.run(unit["argv"],cwd=unit["cwd"],check=True)
link = list(parent["link"])
matches = [i for i,p in enumerate(link) if p.endswith(".o") and (checkout/"build"/p).resolve()==Path(unit["original_object"])]
assert len(matches)==1
link[matches[0]] = unit["object"]
library = stage/"libtorch_npu.so"
builder.replace_flag(link,"-o",library)
subprocess.run(link,cwd=parent["link_cwd"],check=True)
(stage/"receipt.json").write_text(json.dumps(dict(parent=parent["library_sha256"],sha256=sha(library),library=str(library),
    pta=parent["pta_sha"],plugin=parent["op_plugin_sha"],source=sha(dst),object_sha=sha(Path(unit["object"])),
    original=sha(checkout/target),patch_sha=sha(patch),compile=unit,link=link),indent=2))
