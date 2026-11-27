import argparse
from pathlib import Path
import hashlib
import json
import subprocess
import os

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    source=Path(__file__).with_name("batch.cpp")
    root=Path(os.environ.get("ASCEND_HOME_PATH","/usr/local/Ascend/ascend-toolkit/latest")).resolve()
    arch=root/"aarch64-linux"
    if not (arch/"include/aclnn/acl_meta.h").is_file():
        arch=root
    args.out.mkdir(exist_ok=False)
    binary=args.out/"batch.so"
    cmd=["c++","-std=c++17","-O2","-shared","-fPIC","-Wl,-z,defs","-I"+str(arch/"include"),
         str(source),"-L"+str(arch/"lib64"),"-lopapi","-lnnopbase","-o",str(binary)]
    subprocess.run(cmd,check=True)
    (args.out/"build.json").write_text(json.dumps(dict(argv=cmd,sha256=hashlib.sha256(binary.read_bytes()).hexdigest()),indent=2))
