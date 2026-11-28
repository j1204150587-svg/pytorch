"""Compile the pinned storage constructor and substitute it in the verified embedding ELF."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from install import SOURCE,apply

here=Path(__file__).resolve().parent
parent_file=Path("/artifacts/infra-only-20261128/embedding/build/build-receipt.json")
builder_file=Path("/artifacts/dynamic-infra-20261127/build_native.py")


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


parent=json.loads(parent_file.read_bytes())
assert parent["status"]=="built" and parent["library_sha256"]==sha(parent["library"])=="4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"
checkout=Path(parent["base"])/"checkout"
pinned=subprocess.check_output(["git","-C",str(checkout),"show",parent["pta_sha"]+":"+SOURCE])
assert hashlib.sha1(b"blob "+str(len(pinned)).encode()+b"\0"+pinned).hexdigest()=="334ccbf14927fe9b487d570240eacd4901355ce7"
assert (checkout/SOURCE).read_bytes()==pinned
build=Path(parent["link_cwd"])
out=here/"combined"
out.mkdir(exist_ok=False)
staged=out/SOURCE
staged.parent.mkdir(parents=True)
staged.write_text(apply(pinned.decode()))
spec=importlib.util.spec_from_file_location("builder",builder_file)
b=importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
db=json.loads((build/"compile_commands.json").read_bytes())
unit=b.compile_unit(db,checkout,out,SOURCE)
cmd=list(parent["link"])
for obj in parent["compile"]:
    assert sha(obj["object"])==obj["object_sha256"]
idx,=[i for i,s in enumerate(cmd) if s.endswith(".o") and (build/s).resolve()==Path(unit["original_object"])]
cmd[idx]=unit["object"]
lib=out/"libtorch_npu.so"
b.replace_flag(cmd,"-o",lib)
subprocess.run(unit["argv"],cwd=unit["cwd"],check=True)
subprocess.run(cmd,cwd=build,check=True)
receipt={"pta_sha":parent["pta_sha"],"op_plugin_sha":parent["op_plugin_sha"],"parent":str(parent_file),
         "parent_sha256":parent["library_sha256"],"source_blob":"334ccbf14927fe9b487d570240eacd4901355ce7",
         "source_sha256":sha(checkout/SOURCE),"patched_source_sha256":sha(staged),"compile":unit,
         "object_sha256":sha(unit["object"]),"link":cmd,"link_cwd":str(build),"library":str(lib),
         "library_sha256":sha(lib),"status":"built"}
(out/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
print(receipt["library_sha256"])
