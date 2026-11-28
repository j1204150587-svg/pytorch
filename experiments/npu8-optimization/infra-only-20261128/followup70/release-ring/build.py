"""Replace one PTA object in the previously verified bidirectional embedding build."""
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

patch, target, blob = (sys.argv[1:] or [str(Path(__file__).with_name("release-ring.patch")),
                      "torch_npu/csrc/core/npu/NPUQueue.cpp", "5afef4e0ab9ad0c2b5af0fdc56561a5df0c06435"])
patch = Path(patch).resolve()
here = patch.parent
study = here.parents[1]
parent = json.loads((study / "embedding/build/build-receipt.json").read_bytes())
assert parent["library_sha256"] == "4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"
spec = importlib.util.spec_from_file_location("builder", study.parent / "dynamic-infra-20261127/build_native.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
checkout = Path(parent["base"]) / "checkout"
build = checkout / "build"
stage = here / "build"
stage.mkdir(exist_ok=False)
original = (checkout / target).read_bytes()
assert hashlib.sha1(b"blob " + str(len(original)).encode() + b"\0" + original).hexdigest() == blob
assert builder.digest(Path(parent["library"])) == parent["library_sha256"]
for unit in parent["compile"]:
    assert builder.digest(Path(unit["object"])) == unit["object_sha256"]
(stage / target).parent.mkdir(parents=True)
shutil.copyfile(checkout / target, stage / target)
subprocess.run(["patch", "-p1", "-i", str(patch)], cwd=stage, check=True)
database = json.loads((build / "compile_commands.json").read_bytes())
unit = builder.compile_unit(database, checkout, stage, target)
link = list(parent["link"])
library = stage / "libtorch_npu.so"
builder.replace_flag(link, "-o", library)
matches = [i for i, arg in enumerate(link) if arg.endswith(".o") and (build / arg).resolve() == Path(unit["original_object"])]
assert len(matches) == 1
link[matches[0]] = unit["object"]
subprocess.run(unit["argv"], cwd=unit["cwd"], check=True)
subprocess.run(link, cwd=build, check=True)
receipt = dict(parent=parent["library_sha256"], original=builder.digest(checkout / target),
               patch=builder.digest(patch), compile=unit,
               link=link, sha256=builder.digest(library), library=str(library))
(stage / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
