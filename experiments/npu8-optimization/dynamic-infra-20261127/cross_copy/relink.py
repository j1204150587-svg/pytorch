"""Rebuild changed PTA translation units with the historical build commands in an isolated package."""
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

root = Path(__file__).resolve().parent
old = Path(sys.argv[1]).resolve()
checkout = old / "checkout"
build = checkout / "build"
patch = Path(sys.argv[2]).resolve()
stage = root / "patched"
stage.mkdir(exist_ok=True)
library = "packages/torch_npu/lib/libtorch_npu.so"
destination = root / "package/torch_npu/lib/libtorch_npu.so"
plugin = patch.name == "install.py"
if plugin:
    base = "third_party/op-plugin/op_plugin/ops/opapi/"
    changes = [base + name for name in ("CatKernelNpuOpApi.cpp","StructKernelNpuOpApi.cpp")]
else:
    changes = [line[6:] for line in patch.read_text().splitlines() if line.startswith("+++ b/")]
assert changes and len(set(changes)) == len(changes)
assert subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip() == "38e483ee79ec8a3286517808740a811c0f0ae66c"
def run(argv, cwd):
    subprocess.run(argv, cwd=cwd, check=True)

original_digests = {}
for path in changes:
    assert path.endswith((".cpp", ".h")) and path.startswith(("torch_npu/","third_party/op-plugin/")) and ".." not in Path(path).parts
    target = stage / path
    target.parent.mkdir(parents=True, exist_ok=True)
    if plugin:
        shutil.copy2(checkout / path,target)
        original_digests[path] = hashlib.sha256(target.read_bytes()).hexdigest()
    elif (checkout / path).is_file():
        target.write_bytes(subprocess.check_output(["git", "-C", str(checkout), "show", f"HEAD:{path}"]))
if plugin:
    plugin_root=Path("third_party/op-plugin")
    assert subprocess.check_output(["git","-C",str(checkout/plugin_root),"rev-parse","HEAD"],text=True).strip()=="eaf6b55812a1998a4a40e485e3b2861a40e9c2f6"
    header=plugin_root/"op_plugin/utils/op_api_common.h"
    (stage/header).parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(checkout/header,stage/header)
    run([sys.executable,"-B",str(patch),str(stage/plugin_root)],stage)
else:
    run(["git", "apply", "--no-index", str(patch)], stage)
commands = json.loads((build / "compile_commands.json").read_text())
replacements = {}
additional = []
receipt = {"base": str(checkout), "patch_sha256": hashlib.sha256(patch.read_bytes()).hexdigest(), "compile": []}
if plugin:
    receipt["original_sources_sha256"]=original_digests
    receipt["private_header_sha256"]=hashlib.sha256(patch.with_name("list_key45.h").read_bytes()).hexdigest()
for index, path in enumerate(p for p in changes if p.endswith(".cpp")):
    existing = (checkout / path).is_file()
    source = checkout / path if existing else checkout / "torch_npu/csrc/aten/ops/FlattenDenseTensorsKernelNpu.cpp"
    row, = (r for r in commands if r["file"] == str(source))
    argv = shlex.split(row["command"])
    argv.insert(1, "-I" + str(stage))
    obj = argv[argv.index("-o") + 1]
    newobj = str(stage / f"changed-{index}.o")
    if existing:
        replacements[obj] = newobj
    else:
        additional.append(newobj)
    argv[argv.index("-o") + 1] = newobj
    argv[argv.index("-c") + 1] = str(stage / path)
    argv.extend(["-iquote", str((checkout / path).parent)])
    run(argv, build)
    receipt["compile"].append(argv)

link = subprocess.check_output(["ninja", "-t", "commands", library], cwd=build, text=True).splitlines()[-1]
assert link.startswith(": && ") and link.endswith(" && :")
# Parse Ninja's shell-quoted $ORIGIN as the literal linker argument.
argv = shlex.split(link[5:-5])
argv = [arg.replace("\\$ORIGIN", "$ORIGIN") for arg in argv]
argv[1:1] = additional
for obj, newobj in replacements.items():
    assert argv.count(obj) == 1
    argv[argv.index(obj)] = newobj
assert argv[argv.index("-o")+1] == library
temp_library = str(destination.with_suffix(".linking"))
argv[argv.index("-o")+1] = temp_library
run(argv, build)
os.replace(temp_library, destination)
receipt["link"] = argv
receipt["library_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
(root / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
print(receipt["library_sha256"])

