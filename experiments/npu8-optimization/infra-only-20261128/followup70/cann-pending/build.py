"""Compile and relink exactly the public v100 submission TU into an isolated sibling library."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def build(stage,parts):
    root=Path("/study")
    meta=root/"build/compile_commands.json"
    owner="src/runtime/core/src/task/task_submit/v100/task_submit.cc"
    orig=root/"source"/owner
    if sha(orig)!="108d7dbb68562cb1f68324299bd5c7157df7e3f9384243f480950f6e8af17045":raise ValueError(orig)
    atom=root/"source/src/runtime/core/inc/utils/osal.hpp"
    if sha(atom)!="838ee3f0053dd0101f900517394a413ffc5c87d62a820fcbe52dbe3aaf086baf":raise ValueError(atom)
    stage.mkdir(exist_ok=True,parents=True)
    match,=[q for q in json.loads(meta.read_bytes()) if q["file"]==str(orig) and "runtime_v100.dir/" in q["command"]]
    cwd=Path(match["directory"])
    argv=shlex.split(match["command"])
    obj=argv[argv.index("-o")+1]
    source=orig.read_text()
    anchor="    stm->pendingNum_.Add(1U);"
    begin=source.index("rtError_t AllocTaskAndSendStars(")
    index=source.index(anchor,begin)
    if source[:index].count(anchor)!=1:raise ValueError("Incorrect instance")
    modified=(source[:index]+source[index:].replace(anchor,
        "#if defined(__aarch64__) && defined(__linux__)\n    pending_probe::Increment(stm->pendingNum_);\n#else\n"+
        anchor+"\n#endif",1)).replace('namespace cce {',Path(__file__).with_name("pending.inc").read_text()+'\nnamespace cce {',1)
    patched=stage/"task_submit.cc"
    patched.write_text(modified)
    (stage/"candidate.patch").write_text("".join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),
                                        fromfile="a/"+owner,tofile="b/"+owner)))
    newobj=stage/"task_submit.cc.o"
    argv[argv.index("-o")+1]=str(newobj)
    argv[-1]=str(patched)
    if parts in ("compile","all"):
        subprocess.run(argv,cwd=cwd,check=True)
        with (stage/"object.asm").open("w") as f:
            subprocess.run(["objdump","-drC",str(newobj)],stdout=f,check=True)
    if parts=="compile":return
    pristine=cwd/"libruntime_v100.so"
    link=shlex.split((cwd/"CMakeFiles/runtime_v100.dir/link.txt").read_text())
    if link.count(obj)!=1:raise ValueError(obj)
    target=stage/pristine.name
    link[link.index(obj)]=str(newobj)
    link[link.index("-o")+1]=str(target)
    subprocess.run(link,cwd=cwd,check=True)
    def record(path):
        return {"path":str(path),"sha256":sha(path)}
    shipping=Path("/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libruntime_v100.so")
    (stage/"build-receipt.json").write_text(json.dumps(dict(source_sha="0b4af8a31c75a1f1f43638a8f75b9e3c6182e031",
        original=record(orig),atomic=record(atom),modified=record(patched),patch=record(stage/"candidate.patch"),
        pristine_object=record(cwd/obj),modified_object=record(newobj),pristine_library=record(pristine),
        private_library=record(target),shipping=record(shipping),compile=argv,link=link),indent=2))
    with (stage/"candidate.asm").open("w") as f:
        subprocess.run(["objdump","-dC",str(target)],stdout=f,check=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("parts",choices=("compile","link","all"))
    cli.add_argument("--stage",default="/output/source-build",type=Path)
    args=cli.parse_args()
    build(args.stage,args.parts)
