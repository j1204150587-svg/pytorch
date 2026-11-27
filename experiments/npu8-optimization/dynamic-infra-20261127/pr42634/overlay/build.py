"""Build an isolated PR 42634 overlay using the installed TorchNPU and pinned PTA headers."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile

import torch

HEAD = "2701f7b3e8d89625752edbde10e42c2b2172272c"
STOCK = "38e483ee79ec8a3286517808740a811c0f0ae66c"
BUNDLE = "66b90b3359cca24d81a24b4c4256e34a7ae0b928fa53b09100f9bcbf0630a761"
HEADER_BLOBS = {"torch_npu/csrc/framework/OpCommand.h":"687d6a4f3018cc201fb744c5f07b68f14c4c06c0",
                "torch_npu/csrc/inductor/mlir/hacl_rt.h":"86767f9aeea49ade8e066f438c4363c868f76046"}
SOURCES = Path("torch_npu/_inductor/experimental/python_wrapper_fast_launch/csrc")
BINDING_BLOB = "23290b574bb850929a14f279e5764e6e63595574"
SYMS = ("_NPUInductorFastLaunchPlan","_npu_inductor_make_fast_launch_plan","_npu_inductor_fast_launch_with_plan")


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def git_blob(path):
    data = path.read_bytes()
    return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()


def copy_or_link(src,dst):
    try:
        os.link(src,dst)
    except OSError:
        shutil.copy2(src,dst)
    return dst


def includes(checkout):
    commands = json.loads((checkout/"build/compile_commands.json").read_text())
    match = next(x for x in commands if x["file"].endswith("/csrc/npu/Module.cpp"))
    args = match.get("arguments") or shlex.split(match["command"])
    result = []
    for i,opt in enumerate(args):
        if opt in ("-I","-isystem","-iquote"):
            result.extend([opt,args[i+1]])
        elif opt.startswith("-I"):
            result.append(opt)
    return result,match["directory"]


def main(args):
    here = Path(__file__).resolve().parent
    checkout = args.checkout.resolve(strict=True)
    out = args.output.resolve()
    stock = ((args.base_package / "torch_npu") if args.base_package else
             Path(next(iter(importlib.util.find_spec("torch_npu").submodule_search_locations)))).resolve(strict=True)
    torch_root = Path(torch.__file__).resolve().parent
    bundle = here/"pr-source.tar.gz"
    if sha(bundle) != BUNDLE:
        raise ValueError("PR source archive SHA256 differs")
    if out == stock or out.is_relative_to(stock) or (out/"package").exists():
        raise ValueError("select a new output directory outside the installed package")
    out.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="build-",dir=out))
    src = work/"pr-src"
    with tarfile.open(bundle) as tar:
        tar.extractall(src, filter="data")
    cpp = src/SOURCES/"bindings.cpp"
    if git_blob(cpp) != BINDING_BLOB:
        raise ValueError("PR bindings blob differs")
    for name,expected in HEADER_BLOBS.items():
        if git_blob(checkout/name) != expected:
            raise ValueError("pinned stock header differs: " + name)
    args_include,cwd = includes(checkout)
    stage = work/"package"
    target = stage/"torch_npu"
    # Keep the installed _C, libraries and torch_npu bootstrap. The compiler
    # frontend comes from one exact commit, including its package layout.
    shutil.copytree(stock,target,copy_function=copy_or_link,symlinks=True,
                    ignore=lambda p,n: [v for v in n if v=="__pycache__" or (Path(p)==stock and v=="_inductor")])
    shutil.copytree(src/"torch_npu/_inductor",target/"_inductor")
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    binary = target/("_pr42634_register"+suffix)
    abi = int(torch._C._GLIBCXX_USE_CXX11_ABI)
    extra_libs = [Path(v) for v in os.environ.get("LD_LIBRARY_PATH","").split(os.pathsep) if v]
    extra_libs += [Path(v)/p for v in filter(None,(os.environ.get("ASCEND_HOME_PATH"),
                    os.environ.get("ASCEND_TOOLKIT_HOME"),"/usr/local/Ascend/ascend-toolkit/latest",
                    "/usr/local/Ascend/cann")) for p in ("lib64","runtime/lib64","aarch64-linux/lib64")]
    runtime = next((p/"libruntime.so" for p in extra_libs if (p/"libruntime.so").is_file()
                    and "stub" not in str(p).lower()),None)
    if runtime is None:
        raise FileNotFoundError("need CANN's real libruntime.so in LD_LIBRARY_PATH")
    command = [os.environ.get("CXX","c++"),"-std=c++17","-O2","-fPIC","-shared","-fvisibility=hidden",
               f"-D_GLIBCXX_USE_CXX11_ABI={abi}","-I"+str(src),"-I"+str(checkout),
               "-I"+sysconfig.get_paths()["include"],"-I"+str(torch_root/"include"),
               "-I"+str(torch_root/"include/torch/csrc/api/include"),*args_include,
               str(here/"register.cpp"),str(cpp),"-L"+str(stock/"lib"),"-L"+str(torch_root/"lib"),
               "-ltorch_npu","-ltorch_python","-ltorch_cpu","-ltorch","-lc10",
               "-Wl,--no-as-needed",str(runtime),"-Wl,--as-needed","-Wl,-rpath,$ORIGIN/lib",
               "-o",str(binary)]
    subprocess.run(command,cwd=cwd,check=True)
    init = target/"__init__.py"
    before = init.read_bytes()
    tail = b"""\n# Register only absent PR 42634 bindings in the original torch_npu._C namespace.
_pr42634_names = ("_NPUInductorFastLaunchPlan","_npu_inductor_make_fast_launch_plan","_npu_inductor_fast_launch_with_plan")
_pr42634_status = tuple(hasattr(_C, n) for n in _pr42634_names)
if not any(_pr42634_status):
    from . import _pr42634_register
    _pr42634_register.install(_C)
elif not all(_pr42634_status):
    raise RuntimeError("partial native PR 42634 bindings")
"""
    replacement = target/"_pr42634_init.tmp"
    replacement.write_bytes(before+tail)
    replacement.replace(init)
    if (stock/"__init__.py").read_bytes() != before:
        raise RuntimeError("source package unexpectedly changed")
    receipt = {"head":HEAD,"feature":"b1c30ebd929bd381fdb14278657bb7b3da692701","patch_parent":"9412cbac3f7fbf45d24ac2b8fd1a823935bf2e8c",
               "expected_pta_headers_commit":STOCK,"checkout":str(checkout),"header_git_blobs":HEADER_BLOBS,
               "bindings_cpp_git_blob":git_blob(cpp),"archive_sha256":sha(bundle),"stock":str(stock),
               "stock_libtorch_npu_sha256":sha(stock/"lib/libtorch_npu.so"),"torch":torch.__version__,
               "abi":abi,"runtime":str(runtime),"binding_module_sha256":sha(binary),"command":command,
               "original_init_sha256":hashlib.sha256(before).hexdigest(),"overlay_init_sha256":sha(init),
               "overlay_frontend":"exact PR head torch_npu/_inductor; other torch_npu files from installed base"}
    (work/"receipt.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    stage.rename(out/"package")
    (work/"receipt.json").replace(out/"receipt.json")
    print(out/"package",flush=True)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--checkout",type=Path,default=Path("/artifacts/pta08-queued-replay-20261125-C6eCXV/checkout"))
    cli.add_argument("--base-package",type=Path,help="parent of torch_npu; defaults to current installed wheel")
    cli.add_argument("--output",type=Path,required=True)
    main(cli.parse_args())
