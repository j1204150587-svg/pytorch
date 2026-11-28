"""Build the release micro with the exact target compile flags and pinned NPUDefine.cpp."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import sysconfig


here=Path(__file__).resolve().parent
checkout=Path("/artifacts/pta08-queued-replay-20261125-C6eCXV/checkout")
owner="torch_npu/csrc/framework/NPUDefine.cpp"
sha="38e483ee79ec8a3286517808740a811c0f0ae66c"
text=subprocess.check_output(["git","-C",checkout,"show",sha+":"+owner])
assert hashlib.sha1(b"blob "+str(len(text)).encode()+b"\0"+text).hexdigest()=="bd21c142a8e5b5ee7a88be64c038f882801d08fa"
assert (checkout/owner).read_bytes()==text
records=json.loads((checkout/"build/compile_commands.json").read_bytes())
targets=[rec for rec in records if Path(rec["file"])==checkout/owner]
assert len(targets)==1
row=targets[0]
commands=[]
for source,output in ((checkout/owner,"define.o"),(here/"release_cpu.cpp","cpu.o"),(here/"move_only.cpp","move.o")):
    cmd=shlex.split(row["command"])
    cmd[cmd.index("-c")+1]=str(source)
    cmd[cmd.index("-o")+1]=str(here/output)
    if "-MF" in cmd: cmd[cmd.index("-MF")+1]=str(here/(output+".d"))
    cmd[1:1]=["-ffunction-sections","-fdata-sections"]
    commands.append(cmd)
    subprocess.run(cmd,cwd=row["directory"],check=True)
lib=Path(sysconfig.get_paths()["purelib"])/"torch/lib"
exe=here/"release_cpu"
cmd=[cmd[0],"-Wl,--gc-sections",str(here/"cpu.o"),str(here/"move.o"),str(here/"define.o"),
     "-L"+str(lib),"-Wl,-rpath,"+str(lib),"-ltorch_cpu","-lc10","-pthread","-o",str(exe)]
commands.append(cmd)
subprocess.run(cmd,check=True)
(here/"cpu-receipt.json").write_text(json.dumps(dict(pta_sha=sha,blob="bd21c142a8e5b5ee7a88be64c038f882801d08fa",
    commands=commands,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()),indent=2)+"\n")
if "--only-build" not in sys.argv:
    with (here/"cpu.csv").open("w") as f:subprocess.run([exe,here/"request-names.txt"],stdout=f,check=True)
