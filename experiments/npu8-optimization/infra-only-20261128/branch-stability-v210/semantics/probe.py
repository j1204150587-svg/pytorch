"""Launch one existing diagnostic unchanged with the three branch switches fixed before import."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

STUDY=Path("/artifacts/infra-only-20261128")
SOURCES={
    "embedding":("embedding/micro.py","960eabbd63fe9bd87865a0b1a2b5934e0448317a724c0484b9b6bb1aaf56c7bf"),
    "overflow":("followup70-20261128/overflow/check.py","39236be4daa59dd850ab75943edd3b55c8bee54b6038d70dd9e65067dd8b0d9c")}
COUNTER=STUDY/"counter_embedding.so"
COUNTER_SHA="5528151fd5388941af47b417b50b7ced00d2dc661d93425d6008d4d53fb228a7"
SWITCHES=("NPU_DYNAMIC_REBASE_INDICES","NPU_DYNAMIC_REBASE_FORWARD","TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY")
HEAD="e3cda6399cb194c813530222478ea018f5aca5c9"

def digest(path):
    with path.open("rb") as f: return hashlib.file_digest(f,"sha256").hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kind",choices=(*SOURCES,"both"))
    ap.add_argument("arm",choices=("A","B"))
    ap.add_argument("workspace",type=Path)
    ap.add_argument("out",type=Path)
    ap.add_argument("package",type=Path)
    ap.add_argument("sha256")
    args=ap.parse_args()
    if args.kind=="both":
        for kind in SOURCES:
            child=args.out/kind
            child.mkdir(exist_ok=False)
            command=[sys.executable,"-B",str(Path(__file__).resolve()),kind,args.arm,str(args.workspace),str(child),str(args.package),args.sha256]
            (child/"command.json").write_text(json.dumps(command,indent=2)+"\n")
            with (child/"run.log").open("wb") as report:
                rc=subprocess.run(command,stdout=report,stderr=subprocess.STDOUT).returncode
            (child/"exit").write_text(str(rc)+"\n")
            if rc: raise RuntimeError(f"{kind} on {args.arm} exited {rc}; see {child}/run.log")
        return
    relative,expected=SOURCES[args.kind]
    original=STUDY/relative
    lib=args.package/"torch_npu/lib/libtorch_npu.so"
    assert digest(original)==expected and digest(lib)==args.sha256 and digest(COUNTER)==COUNTER_SHA
    assert str(args.package) in os.environ["PYTHONPATH"].split(":") and os.environ["TASK_QUEUE_ENABLE"]=="1"
    value=str(int(args.arm=="B"))
    fixed={k:value for k in SWITCHES}
    fixed.update(NPU_DYNAMIC_REBASE_COUNT="1",TORCH_NPU_QUEUE_WAKE_AUDIT="1",ACLNN_CACHE_LIMIT="65536")
    assert not os.environ.get("LD_PRELOAD"),"unexpected inherited diagnostic library"
    env=dict(os.environ,**fixed,LD_PRELOAD=str(COUNTER),INFRA_COUNTER_DIR=str(args.out))
    meta=dict(target_head=HEAD,package=str(args.package),library=str(lib),sha256=args.sha256,pid=os.getpid(),
              original=str(original),original_sha256=expected,settings=fixed,counter_sha256=COUNTER_SHA,arm=args.arm)
    (args.out/"receipt.json").write_text(json.dumps(meta,indent=2)+"\n")
    options=[str(args.out/"result.json")]
    if args.kind=="overflow":
        # The original checker compares its libtorch_npu mappings against this one library.
        recipe=dict(common={k:v for k,v in fixed.items() if k!=SWITCHES[-1]},switch=SWITCHES[-1],A="0",B="1",
                    libraries={str(lib):args.sha256})
        recipe_path=args.out/"probe-recipe.json"
        recipe_path.write_text(json.dumps(recipe,indent=2)+"\n")
        options=[str(args.workspace),str(args.out/"result.json"),str(recipe_path),args.arm]
    os.execve(sys.executable,[sys.executable,"-B",str(original),*options],env)

if __name__=="__main__":main()

