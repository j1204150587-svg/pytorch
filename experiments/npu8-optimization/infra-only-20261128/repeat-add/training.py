"""Run unchanged eager training after verifying the operator micro gate and its same-stream work."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys


HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
NAME=HERE.name


def read(path):
    return json.loads(path.read_bytes())


def run(mode,order):
    out=STUDY/"pairs"/f"{NAME}-{mode}-{order}"
    if not out.exists():
        recipe=STUDY/"recipes"/(NAME+("-audit" if mode=="audit" else "")+".json")
        with (HERE/f"training-{mode}-{order}.log").open("wb") as log:
            subprocess.run([sys.executable,"-B",STUDY/"runner/host_pair.py","--runner",STUDY/"archived-runner",
                            "--recipe",recipe,"--out",out,"--mode",mode,"--order",order],stdout=log,stderr=subprocess.STDOUT,check=True)
    r=read(out/"result.json")
    if r["exit"]:
        raise RuntimeError(out)
    return r


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("micro_dir",type=Path)
    parser.add_argument("stage",choices=("audit","timing","l0","l1"))
    args=parser.parse_args()
    micro=read(args.micro_dir/"micro.json")
    markers=read(args.micro_dir/"profile-analysis.json")
    if (not micro.get("complete") or any(e["rc"] for e in micro["cases"]) or
        len(markers)!=4 or any(e["launches"]!=8 or e["work"]!={"aclnnAdd_AddAiCore_Add":8} for e in markers)
        or len({e["stream"] for e in markers})!=1):
        raise ValueError("Missing successful micro or physical-stream evidence")
    t=micro["cases"]
    if all(t[b]["median_us"][key]>=t[a]["median_us"][key]
           for a,b in ((2,3),(5,4)) for key in ("shape4","shape5","all60")):
        raise ValueError("Paired regression for all workload shapes and both orders")
    if args.stage=="audit":
        for diagnosis in (False,True):
            recipe=STUDY/"recipes"/(NAME+("-audit" if diagnosis else "")+".json")
            r=read(recipe)
            r["common"].update(ACLNN_CACHE_LIMIT="65536",NPU_DYNAMIC_REBASE_INDICES="1",NPU_DYNAMIC_REBASE_FORWARD="1",
                               NPU_DYNAMIC_REBASE_COUNT="0",NPU_HSTU_REPEAT_ADD_COUNT=str(int(diagnosis)))
            recipe.write_text(json.dumps(r,indent=2)+"\n")
        r=run("audit","ab")
        log=(Path(r["directory"])/"02-B/run.log").read_text()
        match=re.search(r"^REPEAT_ADD (\d+(?: \d+){8})$",log,re.M)
        if not (match and r["audit_passed"]):
            raise ValueError("Training semantics or counts missing")
        counts=list(map(int,match[1].split()))
        accept,_,build,reuse,work,refused,fail,_,single=counts
        if not (build and reuse and accept==work==build+reuse and accept>=8*1023 and refused==fail==single==0):
            raise ValueError(("No full side-table coverage",counts))
        (HERE/"training-gate.json").write_text(json.dumps(dict(counts=counts,result=r),indent=2)+"\n")
    else:
        gate=read(HERE/"training-gate.json")
        if not gate["result"]["audit_passed"]:
            raise ValueError("The unchanged standard 8-step training gate is required")
        for order in (("ab",) if args.stage=="l1" else ("ab","ba")):
            run("no-profiler" if args.stage=="timing" else args.stage,order)
