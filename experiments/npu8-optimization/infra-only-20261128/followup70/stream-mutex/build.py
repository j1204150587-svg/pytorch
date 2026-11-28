"""Relink only the pristine v100 Stream factory translation unit in a private overlay."""
import difflib
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


root=Path("/study")
here=Path("/output")
owner="src/runtime/core/src/stream/v100/stream_creator_c.cc"
original=root/"source"/owner
assert sha(original)=="dc28bcbd430dfef168c740fb23baf4d61b7a9ec2d4280a9e4f99dc6bcc900f35"
stage=here/"private"
stage.mkdir(exist_ok=True)
source=original.read_text()
anchor="    return new (std::nothrow) Stream(dev, prio, stmFlags, dvppGrp);"
assert source.count(anchor)==1
body="""    auto finish = [](Stream *s) {
        if (s != nullptr) {
            const auto serial = layoutCounters[0].fetch_add(1, std::memory_order_relaxed);
            observedStreams[serial % 128].store(reinterpret_cast<uintptr_t>(s), std::memory_order_relaxed);
            observedHandles[serial % 128].store(reinterpret_cast<uintptr_t>(s->GetInnerHandle()), std::memory_order_release);
            if (splitHotFields(s)) layoutCounters[1].fetch_add(1, std::memory_order_relaxed);
        }
        return s;
    };
    if (!splitRequested()) return finish(new (std::nothrow) Stream(dev, prio, stmFlags, dvppGrp));
    layoutCounters[2].fetch_add(1, std::memory_order_relaxed);
    std::array<RawBlock, 12> blocks;
    size_t selected = 0;
    for (size_t i = 0; i < blocks.size(); ++i) {
        blocks[i].reset(::operator new(sizeof(Stream), std::nothrow));
        if (!blocks[i]) break;
        layoutCounters[3].fetch_add(1, std::memory_order_relaxed);
        if (splitHotFields(blocks[i].get())) {
            selected = i;
            break;
        }
    }
    if (!blocks[selected]) return nullptr;
    if (!splitHotFields(blocks[selected].get())) layoutCounters[4].fetch_add(1, std::memory_order_relaxed);
    Stream *const s = ::new(blocks[selected].get()) Stream(dev, prio, stmFlags, dvppGrp);
    // On constructor exceptions all raw allocations are owned by their guards.
    // An ordinary virtual delete can free the selected global-operator-new allocation.
    (void)blocks[selected].release();
    return finish(s);"""
modified=source.replace(anchor,body).replace("namespace cce {",(here/"placement.inc").read_text()+"\nnamespace cce {",1)
newsrc=stage/original.name
newsrc.write_text(modified)
patch=here/"stream-placement.patch"
patch.write_text("".join(difflib.unified_diff(source.splitlines(True),modified.splitlines(True),fromfile="a/"+owner,tofile="b/"+owner)))
match,=[row for row in json.loads((root/"build/compile_commands.json").read_bytes())
        if row["file"]==str(original) and "runtime_v100.dir/" in row["command"]]
cwd=Path(match["directory"])
argv=shlex.split(match["command"])
slot=argv.index("-o")+1
oldobj=argv[slot]
newobj=stage/(original.name+".o")
argv[slot]=str(newobj)
argv[-1]=str(newsrc)
subprocess.run(argv,cwd=cwd,check=True)
target=stage/"libruntime_v100.so"
link=shlex.split((cwd/"CMakeFiles/runtime_v100.dir/link.txt").read_text())
assert link.count(oldobj)==1
link[link.index(oldobj)]=str(newobj)
link[link.index("-o")+1]=str(target)
subprocess.run(link,cwd=cwd,check=True)
shipping=Path("/usr/local/Ascend/cann-9.1.0/aarch64-linux/lib64/libruntime_v100.so")
assert sha(shipping)=="ab4b573989e6b76e129204cf329bdd916739bdd16092a0efcfa88d8ba011d689"
paths=[original,newsrc,patch,cwd/oldobj,newobj,target,cwd/target.name,shipping]
(here/"build-receipt.json").write_text(json.dumps(dict(source_commit="0b4af8a31c75a1f1f43638a8f75b9e3c6182e031",
    sha256={str(p):sha(p) for p in paths},compile=argv,link=link),indent=2)+"\n")
