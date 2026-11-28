"""CPU-only replacement of the context-state TU on the verified dual embedding parent."""
from difflib import unified_diff
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
from apply import patch


def digest(path):
    with path.open("rb") as src:return hashlib.file_digest(src,"sha256").hexdigest()


def main():
    here=Path(__file__).resolve().parent
    parent_file=here.parent/"embedding/build/build-receipt.json"
    parent=json.loads(parent_file.read_bytes())
    assert parent["library_sha256"]==digest(Path(parent["library"]))=="4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"
    checkout=Path(parent["base"])/"checkout"
    build=checkout/"build"
    name="torch_npu/csrc/core/npu/NPUFunctions.cpp"
    src=(checkout/name).read_text()
    out=here/"build"
    out.mkdir(exist_ok=True)
    stage=Path(tempfile.mkdtemp(prefix="compile-",dir=out))
    dest=stage/name
    dest.parent.mkdir(parents=True)
    dest.write_text(patch(src))
    (here/"context-state.patch").write_text("diff --git a/"+name+" b/"+name+"\n"+
        "".join(unified_diff(src.splitlines(True),dest.read_text().splitlines(True),fromfile="a/"+name,tofile="b/"+name)))
    spec=importlib.util.spec_from_file_location("build_native","/artifacts/dynamic-infra-20261127/build_native.py")
    builder=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    obj=builder.compile_unit(json.loads((build/"compile_commands.json").read_bytes()),checkout,stage,name)
    with (stage/"compile.log").open("wb") as log:
        subprocess.run(obj["argv"],cwd=obj["cwd"],stdout=log,stderr=subprocess.STDOUT,check=True)
    obj["object_sha256"]=digest(Path(obj["object"]))
    lib=out/"libtorch_npu.so"
    link=list(parent["link"])
    builder.replace_flag(link,"-o",lib)
    i,=[i for i,arg in enumerate(link) if arg.endswith(".o") and (build/arg).resolve()==Path(obj["original_object"])]
    link[i]=obj["object"]
    subprocess.run(link,cwd=build,check=True)
    receipt=dict(status="built",library=str(lib),library_sha256=digest(lib),parent_receipt=str(parent_file),parent_sha256=parent["library_sha256"],
        stage=str(stage),src=name,original_sha256=digest(checkout/name),patched_sha256=digest(dest),compile=obj,link=link)
    (out/"build-receipt.json").write_text(json.dumps(receipt,indent=2)+"\n")
    print(receipt["library_sha256"])


if __name__=="__main__":
    main()
