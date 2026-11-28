"""Recover CMake source, rebuilt object and reused object evidence from the archived Ninja link."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

here=Path(__file__).resolve().parent
src=here/"cache-worktree"
seed=Path("/data/recsys-gr-ds019-host88/artifacts/pta-build-UK5IES/checkout")
inner=Path("/artifacts/pta-build-UK5IES/checkout")
head="e3cda6399cb194c813530222478ea018f5aca5c9"
base="38e483ee79ec8a3286517808740a811c0f0ae66c"
def sha(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()
def cmd(*args,cwd=None):
    return subprocess.check_output(args,cwd=cwd,text=True).strip()
assert cmd("git","rev-parse","HEAD",cwd=src)==head
assert cmd("git","rev-parse","HEAD",cwd=seed)==base
log=(here/"compile-link.log").read_text()
lines=[x for x in log.splitlines() if re.match(r"^\[\d+/\d+\]",x)]
compile_lines=[l for l in lines if " -c " in l and " -o " in l]
link,=[l for l in lines if " -shared " in l and "libtorch_npu.so" in l]
targets={}
for l in compile_lines:
    args=shlex.split(l)
    filename=args[args.index("-c")+1]
    obj=args[args.index("-o")+1]
    source=src/Path(filename).relative_to(inner)
    targets[obj]={"source":str(source.relative_to(src)),"source_sha256":sha(source),"object_sha256":sha(src/"build"/obj)}
required={"NPUQueue.cpp","EmbeddingKernelNpuOpApi.cpp","StructKernelNpuOpApi.cpp","op_api_common.cpp","OpCommand.cpp"}
assert {Path(v["source"]).name for v in targets.values()}==required
objects=[x for x in shlex.split(link) if x.endswith(".o")]
unchanged=[]
for obj in objects:
    if obj in targets: continue
    candidate=src/"build"/obj
    reference=seed/"build"/obj
    value=sha(candidate)
    assert value==sha(reference),obj
    unchanged.append({"file":obj,"sha256":value})
assert len(objects)>100 and len(unchanged)+5==len(objects)
(here/"reused-objects.json").write_text(json.dumps(unchanged,indent=2)+"\n")
(here/"link-command.txt").write_text(link+"\n")
containers=[]
for name in ["recsys-dynbranch-e3cda63-build","recsys-dynbranch-e3cda63-package2"]:
    data=json.loads(cmd("docker","inspect",name))[0]
    containers.append({"name":name,"image":data["Image"],"started":data["State"]["StartedAt"],
        "finished":data["State"]["FinishedAt"],"exit":data["State"]["ExitCode"],"runtime":data["HostConfig"]["Runtime"],
        "devices":data["HostConfig"]["Devices"],"network":data["HostConfig"]["NetworkMode"],
        "binds":data["HostConfig"]["Binds"],"entrypoint":data["Config"]["Entrypoint"],"cmd":data["Config"]["Cmd"]})
assert all(c["exit"]==0 and c["runtime"]=="runc" and not c["devices"] for c in containers)
patched=sorted((src/"third_party/op-plugin-patches").rglob("*"))
r=dict(commit=head,base=base,op_plugin=cmd("git","rev-parse","HEAD",cwd=src/"third_party/op-plugin"),
    branch_changes=cmd("git","diff","--name-only",base,head,cwd=src).splitlines(),
    top_level_cmake_sha256=sha(src/"CMakeLists.txt"),
    patch_hashes={f.relative_to(src).as_posix():sha(f) for f in patched if f.is_file()},
    rebuilt_objects=targets,reused_sha256_identical=len(unchanged),link_total_objects=len(objects),
    linker_evidence_sha256=sha(here/"link-command.txt"),reused_list_sha256=sha(here/"reused-objects.json"),
    final_linker_elf_sha256=sha(here/"libtorch_npu.so"),cmake_output_sha256=sha(src/"build/packages/torch_npu/lib/libtorch_npu.so"),
    toolchain=cmd("readelf","-p",".comment",str(here/"libtorch_npu.so")),
    elf_dynamic=cmd("readelf","-d",str(here/"libtorch_npu.so")),containers=containers)
assert r['final_linker_elf_sha256']==r['cmake_output_sha256']
(here/"build-provenance.json").write_text(json.dumps(r,indent=2)+"\n")
print(r["reused_sha256_identical"],r["link_total_objects"],r["final_linker_elf_sha256"])
