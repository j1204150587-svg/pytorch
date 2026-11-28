"""Build the CANN Host route against the exact libraries and operator installed in the training image."""
import hashlib
import json
from pathlib import Path
import subprocess

here = Path(__file__).resolve().parent
cann = Path("/usr/local/Ascend/cann-9.1.0")
platform = cann/"aarch64-linux"
library = here/"private-host.so"
target = "EmbeddingDenseGrad_bf16_high_performance"
kernel = cann/"opp/built-in/op_impl/ai_core/tbe/kernel/ascend910_93/ops_nn/embedding_dense_grad_v2"
code = cann/"opp/built-in/op_impl/ai_core/tbe/impl/ops_nn/ascendc/embedding_dense_grad_v2"
cmd = ["g++", "-std=c++17", "-O2", "-fPIC", "-shared", "-D_GLIBCXX_USE_CXX11_ABI=0", str(here/"host.cpp"),
       *["-I"+str(platform/p) for p in ("include","include/aclnn","include/experiment","include/experiment/platform")],
       "-L"+str(platform/"lib64"), "-Wl,-z,defs", "-lopapi", "-lnnopbase", "-lascendcl",
       "-lgraph", "-lc_sec", "-o", str(library)]
subprocess.run(cmd,check=True)
def digest(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp,"sha256").hexdigest()
files = [here/"host.cpp",library,kernel/(target+".json"),kernel/(target+".o"),
         code/"embedding_dense_grad_v2_small_dim.h",code/"embedding_dense_grad_v2.cpp",
         *[platform/("lib64/"+f) for f in ("libopapi.so","libnnopbase.so","libascendcl.so","libgraph.so","libc_sec.so")]]
(here/"build-receipt.json").write_text(json.dumps(dict(cmd=cmd,files={str(f):digest(f) for f in files}),
                                                  indent=2))
