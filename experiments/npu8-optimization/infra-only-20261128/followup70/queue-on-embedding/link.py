"""Combine the exact queue unit and embedding units that were measured independently."""
import hashlib
import json
from pathlib import Path
import subprocess

here=Path(__file__).resolve().parent
study=here.parents[1]
def sha(p):
    with Path(p).open("rb") as fp:
        return hashlib.file_digest(fp,"sha256").hexdigest()
def get(p,expected):
    data=json.loads(p.read_bytes())
    assert sha(data["library"])==data["library_sha256"]==expected
    for u in data["compile"]:
        assert sha(u["object"])==u["object_sha256"]
    return data
p=study/"embedding/build/build-receipt.json"
q=study/"queue/build-v2/build-receipt.json"
emb=get(p,"4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c")
queue=get(q,"2c3f257319d88b14bd628de41d74dec511ce0e00ea4de14baa64b2d5cd9e2690")
assert all(emb[k]==queue[k] for k in ("pta_sha","op_plugin_sha"))
unit,=queue["compile"]
cmd=list(emb["link"])
cwd=Path(emb["link_cwd"])
idx,=[i for i,v in enumerate(cmd) if v.endswith(".o") and (cwd/v).resolve()==Path(unit["original_object"])]
cmd[idx]=unit["object"]
library=here/"libtorch_npu.so"
cmd[cmd.index("-o")+1]=str(library)
subprocess.run(cmd,cwd=cwd,check=True)
(here/"receipt.json").write_text(json.dumps(dict(library=str(library),sha256=sha(library),
    parents={str(f):sha(f) for f in (p,q)},queue_source=unit["source"],queue_object=unit["object_sha256"],
    pta=emb["pta_sha"],plugin=emb["op_plugin_sha"],cmd=cmd,cwd=str(cwd)),indent=2))
