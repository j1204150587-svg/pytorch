"""Prepare the two native-runtime recipes and invoke the unmodified, serialized HSTU pair driver."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent
STUDY=HERE.parents[1]
INNER="/artifacts/"+STUDY.name+"/followup70/cann-pending"


def read(path):
    return json.loads(path.read_bytes())


def run(stage,order="ab"):
    root=STUDY/"pairs"/f"cann-pending-{stage}-{order}"
    if not root.exists():
        recipe=STUDY/"recipes"/("cann-pending"+("-audit" if stage=="audit" else "")+".json")
        with (HERE/f"training-{stage}-{order}.log").open("wb") as log:
            subprocess.run([sys.executable,"-B",STUDY/"runner/host_pair.py","--runner",STUDY/"archived-runner",
                            "--recipe",recipe,"--out",root,"--mode",stage,"--order",order],stdout=log,stderr=subprocess.STDOUT,check=True)
    r=read(root/"result.json")
    if r["exit"]:raise ValueError((root,r["exit"]))
    return r


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage",choices=("audit","no-profiler","l0","l1"))
    a=cli.parse_args()
    if a.stage=="audit":
        m=read(HERE/"probe1/analysis.json")
        if any(v["real_acl_calls"]!=32 for v in m["profile"]) or len(m["profile"])!=4:raise ValueError("Missing measured launch counts")
        common=read(STUDY/"recipes/emb-fwd.json")
        runtime=INNER+"/source-v2/libruntime_v100.so"
        dump=INNER+"/dump.so"
        for diagnosed in (False,True):
            recipe=dict(common,common=dict(common["common"]),libraries=dict(common["libraries"]))
            recipe.update(switch="NPU_CANN_PENDING_LSE",A="0",B="1")
            recipe["common"].update(NPU_DYNAMIC_REBASE_FORWARD="1",NPU_CANN_PENDING_COUNT=str(int(diagnosed)),LD_PRELOAD=runtime)
            recipe["libraries"][runtime]=read(HERE/"source-v2/build-receipt.json")["private_library"]["sha256"]
            if diagnosed:
                recipe["common"].update(LD_PRELOAD=runtime+":"+dump,INFRA_COUNTER_DIR="{output}")
                recipe["libraries"][dump]=hashlib.sha256((HERE/"dump.so").read_bytes()).hexdigest()
            (STUDY/"recipes"/("cann-pending"+("-audit" if diagnosed else "")+".json")).write_text(json.dumps(recipe,indent=2))
        r=run(a.stage)
        if not r["audit_passed"]:raise ValueError("Original 8-step comparison failed")
        for leg,diagnosis in r["native_counters"].items():
            c=diagnosis["cann-pending"]
            if (c["selected"]!=int(leg=="B") or c["hwcap_atomics"]!=1 or c["mapped"]!=runtime or
                (c["llsc"] if leg=="A" else c["lse"])<8*1023 or (c["lse"] if leg=="A" else c["llsc"])!=0):
                raise ValueError((leg,c))
        (HERE/"training-audit.json").write_text(json.dumps(r,indent=2)+"\n")
    else:
        if not read(HERE/"training-audit.json")["audit_passed"]:raise ValueError("Missing training semantic check")
        for order in (("ab",) if a.stage=="l1" else ("ab","ba")):
            run(a.stage,order)


if __name__=="__main__":main()
