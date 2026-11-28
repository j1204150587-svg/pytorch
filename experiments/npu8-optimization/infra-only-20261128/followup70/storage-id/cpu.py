"""Compile the exact patched source with function sections for a device-free C10 harness."""
import json
import os
from pathlib import Path
import subprocess
import sysconfig

here=Path(__file__).resolve().parent
receipt=json.loads((here/"combined/receipt.json").read_bytes())
row=receipt["compile"]
source=here/"combined/torch_npu/csrc/core/NPUStorageImpl.cpp"


def compile_one(src,output):
    cmd=list(row["argv"])
    cmd[cmd.index("-c")+1]=str(src)
    cmd[cmd.index("-o")+1]=str(output)
    if "-MF" in cmd: cmd[cmd.index("-MF")+1]=str(output.with_suffix(".d"))
    cmd[1:1]=["-ffunction-sections","-fdata-sections"]
    subprocess.run(cmd,cwd=row["cwd"],check=True)


obj=here/"constructor.o"
compile_one(source,obj)
main=here/"fixture.o"
compile_one(here/"storage_cpu.cpp",main)
lib=Path(sysconfig.get_paths()["purelib"])/"torch/lib"
exe=here/"storage_cpu"
subprocess.run([row["argv"][0],"-Wl,--gc-sections",main,obj,"-L"+str(lib),"-Wl,-rpath,"+str(lib),
                "-ltorch_cpu","-lc10","-pthread","-o",exe],check=True)
rows=[]
for mode in ("0","1","1","0")*3:
    env=dict(os.environ,TORCH_NPU_STORAGE_ATOMIC_ID=mode,TORCH_NPU_STORAGE_ID_AUDIT="0")
    rows.append(json.loads(subprocess.check_output([exe,"1","500000"],env=env)))
for lanes in (2,4,8):
    env=dict(os.environ,TORCH_NPU_STORAGE_ATOMIC_ID="1",TORCH_NPU_STORAGE_ID_AUDIT="0")
    rows.append(json.loads(subprocess.check_output([exe,str(lanes),"100000"],env=env)))
(here/"cpu.json").write_text(json.dumps(rows,indent=2)+"\n")
print(json.dumps(rows),flush=True)
