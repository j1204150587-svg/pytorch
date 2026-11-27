"""Compile a standalone dispatcher extension against the running PyTorch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch


def build(out, source=None):
    out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    src=Path(source).resolve() if source else Path(__file__).resolve().with_name("native_rms.cpp")
    base=Path(torch.__file__).resolve().parent
    lib=out/(src.stem+(".dll" if os.name=="nt" else ".so"))
    receipt=out/("rms-build.json" if src.stem=="native_rms" else src.stem+"-build.json")
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    fingerprint={"source_sha256":sha(src),"torch":str(torch.__version__),
                 "abi":int(torch._C._GLIBCXX_USE_CXX11_ABI),"python":sys.version}
    if lib.exists() and receipt.exists():
        old=json.loads(receipt.read_text())
        if all(old.get(k)==v for k,v in fingerprint.items()) and old.get("library_sha256")==sha(lib):
            return lib
    includes=[base/"include",base/"include/torch/csrc/api/include"]
    with tempfile.TemporaryDirectory(dir=out) as tmp:
        dest=Path(tmp)/lib.name
        if os.name=="nt":
            cmd=[os.getenv("CXX","cl"),"/nologo","/std:c++17","/O2","/EHsc","/MD","/bigobj","/LD","/DNOMINMAX",
                 *["/I"+str(x) for x in includes],str(src),"/link","/OUT:"+str(dest),"/LIBPATH:"+str(base/"lib"),
                 "torch_cpu.lib","torch.lib","c10.lib"]
        else:
            cmd=[os.getenv("CXX","c++"),"-std=c++17","-O2","-fPIC","-shared","-fvisibility=hidden",
                 "-D_GLIBCXX_USE_CXX11_ABI="+str(fingerprint["abi"]),*["-I"+str(x) for x in includes],str(src),
                 "-L"+str(base/"lib"),"-ltorch_cpu","-ltorch","-lc10","-o",str(dest)]
        subprocess.run(cmd,cwd=tmp,check=True)
        dest.replace(lib)
    receipt.write_text(json.dumps(dict(fingerprint,argv=cmd,library_sha256=sha(lib)),indent=2)+"\n")
    return lib


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("output",type=Path)
    print(build(p.parse_args().output))
