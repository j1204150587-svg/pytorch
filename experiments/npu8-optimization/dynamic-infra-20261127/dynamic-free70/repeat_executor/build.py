"""Build a standalone probe against the target's own ACLNN operator headers and libraries."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--root",type=Path,default=Path(os.environ.get("ASCEND_HOME_PATH","/usr/local/Ascend/ascend-toolkit/latest")))
    args=parser.parse_args()
    roots=[args.root.resolve()]
    arch=Path(os.uname().machine+"-linux")
    roots.extend([p for p in (args.root/arch,Path("/usr/local/Ascend/cann"),Path("/usr/local/Ascend/ascend-toolkit/latest"))
                  if p.is_dir() and p.resolve() not in roots])
    def find(name):
        for root in roots:
            for d in (root/"include",root/"lib64",root):
                if d.is_dir():
                    for file in d.rglob(name):
                        if file.is_file() and "stub" not in file.parts and "x86_64-linux" not in file.parts:
                            return file.resolve()
        raise FileNotFoundError(name)
    meta=find("acl_meta.h")
    op=find("aclnn_embedding_dense_backward.h")
    libs=[find(k) for k in ("libopapi.so","libnnopbase.so")]
    inc={str(meta.parent.parent),str(op.parent.parent),str(op.parent),str(meta.parent),
         *(str(root/"include") for root in roots),*(str(p) for root in roots for p in (root/arch/"include",))}
    here=Path(__file__).resolve().parent
    args.out.mkdir(parents=True,exist_ok=True)
    output=args.out/"repeat_probe.so"
    if output.exists():
        raise ValueError(f"new build directory required: {output}")
    cmd=[os.environ.get("CXX","c++"),"-std=c++17","-O2","-fPIC","-shared","-Wl,-z,defs",
         *["-I"+i for i in sorted(inc)],"-include",str(meta),"-include",str(op),str(here/"probe.cpp"),
         *[str(lib) for lib in libs],"-o",str(output)]
    subprocess.run(cmd,check=True)
    def digest(p):
        with p.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()
    receipt=dict(argv=cmd,hashes={str(x):digest(x) for x in (meta,op,*libs,here/"probe.cpp",output)})
    (args.out/"build.json").write_text(json.dumps(receipt,indent=2))


if __name__=="__main__":
    main()
