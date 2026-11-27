"""Add the upstream shape-handling bindings to the already-isolated PR package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sysconfig
import tarfile
import tempfile

import torch


EXPECTED = {
    "aoti_package/shape_handling.cpp":"a78d14918d0a989defeda4b39a223bba1fedccc6",
    "aoti_package/shape_handling.h":"ca3ce4068fdb16e8b6f21c6ed73d0317e9d2f002",
    "aoti_torch/npu_shape_handling.cpp":"055c5748b2a019bb4c2f71cf9e2d4c545884acc4",
    "aoti_torch/npu_shape_handling.h":"0ebad8a46308e2ad87cb8357f8636174a3fcbea6"}
TAIL = b"""
# Exact Inductor frontend shape types from PR 42634 head.
_pr42634_shape_names = ("_NPUShapeHandling","ShapeType","ShapePolicy","_ShapeOpStrategyBase","_BSShapeOpStrategy","_SeqShapeOpStrategy")
_pr42634_shape_status = tuple(hasattr(_C,n) for n in _pr42634_shape_names)
if not any(_pr42634_shape_status):
    from . import _pr42634_shape
    _pr42634_shape.install(_C)
elif not all(_pr42634_shape_status):
    raise RuntimeError("partial native PR 42634 shape bindings")
"""


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(out):
    out = out.resolve(strict=True)
    here = Path(__file__).resolve().parent
    r = json.loads((out / "receipt.json").read_text())
    pkg = (out/"package/torch_npu").resolve(strict=True)
    if not pkg.is_relative_to(out) or r["head"] != "2701f7b3e8d89625752edbde10e42c2b2172272c":
        raise ValueError("expected isolated PR overlay")
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    if abi != r["abi"]:
        raise ValueError("PyTorch ABI differs from original overlay")
    src_bundle = here/"shape-source.tar.gz"
    if sha(src_bundle) != "83358751423da90806b960112380d5b60cf192b4ccfa73cbb6f7fbe4cbea9f0e":
        raise ValueError("shape source archive hash differs")
    work = Path(tempfile.mkdtemp(prefix="shape-build-",dir=out))
    with tarfile.open(src_bundle) as t:
        for name,blob in EXPECTED.items():
            path = "torch_npu/csrc/inductor/"+name
            data = t.extractfile(path).read()
            if hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest() != blob:
                raise ValueError("unexpected git blob " + name)
            f = work/path
            f.parent.mkdir(exist_ok=True,parents=True)
            f.write_bytes(data)
    binary = pkg/("_pr42634_shape"+sysconfig.get_config_var("EXT_SUFFIX"))
    init = pkg/"__init__.py"
    if binary.exists() and TAIL in init.read_bytes():
        prev = json.loads((out/"shape-build.json").read_text())
        if prev["binary_sha256"] != sha(binary):
            raise ValueError("existing shape extension hash differs")
        print(binary)
        return
    lib = Path(torch.__file__).resolve().parent
    cc = [os.environ.get("CXX","c++"),"-std=c++17","-O0","-g0","-fPIC","-fvisibility=hidden",
          "-D_GLIBCXX_USE_CXX11_ABI="+str(abi),"-I"+str(work),"-I"+r["checkout"],
          "-I"+sysconfig.get_paths()["include"],"-I"+str(lib/"include"),"-I"+str(lib/"include/torch/csrc/api/include")]
    sources = [here/"shape_register.cpp",work/"torch_npu/csrc/inductor/aoti_package/shape_handling.cpp",
               work/"torch_npu/csrc/inductor/aoti_torch/npu_shape_handling.cpp"]
    commands,objects = [],[]
    for source in sources:
        obj = work/(source.stem+".o")
        command = [*cc,"-c",str(source),"-o",str(obj)]
        commands.append(command)
        subprocess.run(command,cwd=work,check=True)
        objects.append(str(obj))
    built = work/binary.name
    command = [cc[0],"-shared",*objects,"-L"+str(lib/"lib"),"-ltorch_python","-ltorch_cpu","-ltorch","-lc10",
               "-Wl,-rpath,$ORIGIN/lib","-o",str(built)]
    commands.append(command)
    subprocess.run(command,cwd=work,check=True)
    original = init.read_bytes()
    if TAIL in original:
        raise ValueError("bootstrap already contains shape import without its binary")
    updated = work/"__init__.py"
    updated.write_bytes(original+TAIL)
    built.replace(binary)
    updated.replace(init)
    record = {"bindings_from":r["head"],"source_git_blobs":EXPECTED,"source_bundle_sha256":sha(src_bundle),
              "commands":commands,"binary_sha256":sha(binary),"init_sha256":sha(init),"abi":abi}
    (out/"shape-build.json").write_text(json.dumps(record,indent=2)+"\n")
    print(binary,flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("overlay_output",type=Path)
    build(cli.parse_args().overlay_output)
