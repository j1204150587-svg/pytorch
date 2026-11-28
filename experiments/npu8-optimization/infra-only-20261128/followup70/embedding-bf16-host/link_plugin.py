"""Recompile the single embedding entry point on the measured embedding baseline."""
import difflib
import hashlib
import json
from pathlib import Path
import subprocess

here=Path(__file__).resolve().parent
study=here.parents[1]
parent=json.loads((study/"embedding/build/build-receipt.json").read_bytes())
def sha(p):
    with Path(p).open("rb") as fp:
        return hashlib.file_digest(fp,"sha256").hexdigest()
assert sha(parent["library"])==parent["library_sha256"]=="4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"
file="op_plugin/ops/opapi/StructKernelNpuOpApi.cpp"
build=here/"plugin"
source=build/file
source.parent.mkdir(exist_ok=True,parents=True)
original=Path(parent["stage"])/file
assert sha(original)==parent["patched_sources_sha256"][file]
before=original.read_text()
include='#include "op_plugin/utils/dynamic_embedding_scope.h"'
entry="    dynamic_embedding_rebase::Scope scope(indices);"
assert before.count(entry)==before.count(include)==1
after=before.replace(include,include+"\n"+(here/"route.inc").read_text()).replace(entry,entry+"""
    if (selectNativeBf16Embedding(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq)) {
        runNativeBf16Embedding(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out);
        return out;
    }""")
source.write_text(after)
patch=here/"route.patch"
patch.write_text("".join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile="a/"+file,tofile="b/"+file)))
u,=[u for u in parent["compile"] if u["source"]==file]
for item in parent["compile"]:
    assert sha(item["object"])==item["object_sha256"]
obj=build/"entry.o"
cmd=list(u["argv"])
cmd[cmd.index("-c")+1]=str(source)
cmd[cmd.index("-o")+1]=str(obj)
subprocess.run(cmd,cwd=u["cwd"],check=True)
link=list(parent["link"])
link[link.index(u["object"])]=str(obj)
lib=build/"libtorch_npu.so"
link[link.index("-o")+1]=str(lib)
subprocess.run(link,cwd=parent["link_cwd"],check=True)
(build/"receipt.json").write_text(json.dumps(dict(pta=parent["pta_sha"],plugin=parent["op_plugin_sha"],
   parent=parent["library_sha256"],inputs={str(p):sha(p) for p in (original,source,patch,here/"route.inc",here/"private-host.so")},
   compile=cmd,cwd=u["cwd"],link=link,link_cwd=parent["link_cwd"],library=str(lib),sha256=sha(lib)),indent=2))
