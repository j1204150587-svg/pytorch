"""Build a metadata-only Python extension against this interpreter's installed PyTorch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile
import torch


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    source=Path(__file__).resolve().with_name("host_guard.cpp")
    out=args.output.resolve()
    out.mkdir(parents=True,exist_ok=True)
    base=Path(torch.__file__).resolve().parent
    inc=[Path(sysconfig.get_paths()["include"]),base/"include",base/"include/torch/csrc/api/include"]
    suffix=sysconfig.get_config_var("EXT_SUFFIX")
    target=out/("_hstu_host_guard"+suffix)
    check_file=out/"build.json"
    info={"torch":torch.__version__,"python":sys.version,"source_sha256":sha(source),"abi":torch._C._GLIBCXX_USE_CXX11_ABI}
    if check_file.is_file() and target.is_file():
        old=json.loads(check_file.read_text())
        if all(old[k]==v for k,v in info.items()) and sha(target)==old["library_sha256"]:
            print(target);return
    with tempfile.TemporaryDirectory(dir=out) as d:
        built=Path(d)/target.name
        if os.name=="nt":
            cmd=[os.getenv("CXX","cl"),"/nologo","/std:c++17","/O2","/EHsc","/MD","/bigobj","/LD","/DNOMINMAX",
                 *["/I"+str(i) for i in inc],str(source),"/link","/OUT:"+str(built),"/LIBPATH:"+str(base/"lib"),
                 "/LIBPATH:"+str(Path(sys.base_prefix)/"libs"),"torch_python.lib","torch_cpu.lib","torch.lib","c10.lib"]
        else:
            cmd=[os.getenv("CXX","c++"),"-std=c++17","-O2","-shared","-fPIC","-fvisibility=hidden",
                 "-D_GLIBCXX_USE_CXX11_ABI="+str(int(info["abi"])),*["-I"+str(i) for i in inc],str(source),
                 "-L"+str(base/"lib"),"-ltorch_python","-ltorch_cpu","-ltorch","-lc10","-o",str(built)]
        subprocess.run(cmd,cwd=d,check=True)
        built.replace(target)
    info.update(library=target.name,library_sha256=sha(target),command=cmd)
    check_file.write_text(json.dumps(info,indent=2)+"\n",encoding="utf-8")
    print(target,flush=True)


if __name__=="__main__":
    main()
