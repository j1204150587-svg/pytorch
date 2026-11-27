"""Build the PTA sidecar against installed torch; overlay a copy of the exact baseline package."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch
from torch.utils.cpp_extension import include_paths, library_paths


def digest(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-package",type=Path,required=True,help="directory containing the baseline torch_npu package")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    base=args.base_package.resolve(strict=True)
    out=args.output.resolve()
    here=Path(__file__).resolve().parent
    out.mkdir(parents=True,exist_ok=True)
    if (out/"package").exists() or out.is_relative_to(base):
        raise ValueError("need a new output outside the baseline")
    work=Path(tempfile.mkdtemp(prefix="build-",dir=out))
    # The original libtorch_npu.so and every other file are read-only inputs.
    # Use an ordinary copy when a hard link is unsupported.
    def link_or_copy(src,dst):
        try:
            os.link(src,dst)
        except OSError:
            shutil.copy2(src,dst)
        return dst
    package=work/"package"
    shutil.copytree(base/"torch_npu",package/"torch_npu",copy_function=link_or_copy,symlinks=True)
    target=package/"torch_npu"
    extension=target/"dynamic_meta_unflatten.so"
    adapter=target/"_dynamic_meta_unflatten.py"
    if extension.exists() or adapter.exists():
        raise ValueError("baseline already has experimental extension")
    abi=int(torch._C._GLIBCXX_USE_CXX11_ABI)
    command=[os.environ.get("CXX","c++"),"-std=c++17","-O2","-fPIC","-shared",
             f"-D_GLIBCXX_USE_CXX11_ABI={abi}",*["-I"+i for i in include_paths()],
             str(here/"meta_unflatten.cpp"),*["-L"+p for p in library_paths()],
             "-ltorch_cpu","-ltorch","-lc10","-o",str(extension)]
    subprocess.run(command,check=True)
    shutil.copyfile(here/"adapter.py",adapter)
    init=target/"__init__.py"
    original=(base/"torch_npu/__init__.py").read_bytes()
    # Replace the directory entry, never write through a hard link.
    tmp=target/"_dynamic_init.tmp"
    with tmp.open("xb") as f:
        f.write(original+b"\nfrom ._dynamic_meta_unflatten import install as _install_dynamic_meta_unflatten\n_install_dynamic_meta_unflatten()\n")
    tmp.replace(init)
    if (base/"torch_npu/__init__.py").read_bytes()!=original:
        raise ValueError("baseline package changed")
    receipt=dict(torch_version=torch.__version__,base_package=str(base),abi=abi,argv=command,
                 library_sha256=digest(base/"torch_npu/lib/libtorch_npu.so"),
                 original_init_sha256=hashlib.sha256(original).hexdigest(),modified_init_sha256=digest(init),
                 source_sha256=digest(here/"meta_unflatten.cpp"),adapter_sha256=digest(adapter),
                 sidecar_sha256=digest(extension))
    package.rename(out/"package")
    (out/"build-receipt.json").write_text(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
