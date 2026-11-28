"""Launch one attention micro gate under the same card preflight and lock as training."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys

SOURCE=Path(__file__).resolve().parent
FREE70=SOURCE.parent
STUDY=FREE70.parent
sys.path.insert(0,str(STUDY/"runner"))
import run_npu8 as runner


def inner(path):
    return str(Path("/artifacts")/path.resolve().relative_to(STUDY.parent))


def run(output):
    output.mkdir(parents=False,exist_ok=False)
    with runner.LOCK.open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        container,init=runner.inspect_container(STUDY.parent,output)
        runner.check_card(output,"before",container,init)
        argv=[*runner.DOCKER,"exec","--env","TASK_QUEUE_ENABLE=1","--env","PYTHONDONTWRITEBYTECODE=1",
              container,"bash","-c",
              'source /workspace/env.sh && unset ASCEND_RT_VISIBLE_DEVICES && exec python3 -B "$@"',
              "micro",inner(SOURCE/"gate.py"),"--device","npu","--model-dir","/workspace",
              "--library",inner(FREE70/"attention-autograd-build/attention_node.so"),
              "--schedule",inner(SOURCE/"dynamic_shapes_npu8.json"),"--out",inner(output/"micro.json"),
              "--repeats","32"]
        (output/"command.json").write_text(json.dumps(argv,indent=2)+"\n")
        with (output/"run.log").open("wb") as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
        runner.check_card(output,"after",container,init)
        print((output/"run.log").read_text(errors="replace")[-2500:],flush=True)
        result.check_returncode()


if __name__=="__main__":
    run(Path(sys.argv[1]).resolve())
