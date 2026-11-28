"""Gate native lazy initialization, then run the original paired dynamic training."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
INNER = "/artifacts/infra-only-20261128"
NAME = "lazy-gil"


def read(p):
    return json.loads(p.read_bytes())


def run(cmd):
    print(time.strftime("%FT%T%z"), *map(str,cmd),flush=True)
    subprocess.run(list(map(str,cmd)),cwd=ROOT,check=True)


def main():
    link = read(HERE/"combined/receipt.json")
    run([sys.executable,"-B","make_recipe.py",NAME,link["library"],link["library_sha256"],
         "TORCH_NPU_LAZY_GIL_FAST","--diagnostic-switch","TORCH_NPU_LAZY_GIL_AUDIT"])
    for audit in (False,True):
        path=ROOT/"recipes"/(NAME+("-audit" if audit else "")+".json")
        recipe=read(path)
        recipe["common"].update(ACLNN_CACHE_LIMIT="65536",NPU_DYNAMIC_REBASE_INDICES="1",
                                NPU_DYNAMIC_REBASE_FORWARD="1",NPU_DYNAMIC_REBASE_COUNT=str(int(audit)))
        if audit:
            counter=INNER+"/counter_embedding.so"
            recipe["common"].update(LD_PRELOAD=counter,INFRA_COUNTER_DIR="{output}")
            recipe["libraries"][counter]=hashlib.sha256((ROOT/"counter_embedding.so").read_bytes()).hexdigest()
        path.write_text(json.dumps(recipe,indent=2)+"\n")
    for v in ("0","1"):
        out=HERE/("micro-"+v)
        if not (out/"after.card.json").exists():
            run([sys.executable,"-B","operator_probe.py","--package",INNER+"/stock-overlay/"+NAME,
                 "--env","ACLNN_CACHE_LIMIT=65536","--env","NPU_DYNAMIC_REBASE_FORWARD=1","--env",
                 "NPU_DYNAMIC_REBASE_INDICES=1",out,INNER+"/followup70/lazy-gil/micro.py",
                 "@OUT/micro.json",v])
        assert (out/"train.exit").read_text().strip()=="0"
    a,b = (read(HERE/("micro-"+v)/"micro.json") for v in ("0","1"))
    assert len(a)==5 and a==b
    for mode,order in (("audit","ab"),("no-profiler","ab"),("no-profiler","ba"),("l0","ab"),("l0","ba"),("l1","ab")):
        pair=ROOT/"pairs"/f"{NAME}-{mode}-{order}"
        file=pair/"result.json"
        if not file.exists():
            recipe=NAME+("-audit" if mode=="audit" else "")
            run([sys.executable,"-B","runner/host_pair.py","--runner","archived-runner",
                 "--recipe",f"recipes/{recipe}.json","--out",pair,"--mode",mode,"--order",order])
        entry=read(file)
        assert entry["exit"]==0
        if mode=="audit":
            assert entry["audit_passed"] and entry["audited_steps"]==8
            totals={}
            for leg in ("01-A","02-B"):
                raw=Path(entry["directory"])/leg
                matches=re.findall(r"PTA_LAZY_GIL enabled=(\d+) calls=(\d+) fast=(\d+) acquire_ns=(\d+) max_ns=(\d+)",
                                   (raw/"run.log").read_text(errors="replace"))
                record,=matches
                enabled,n,used,ns,max_ns=map(int,record)
                assert enabled==(leg[-1]=="B") and n>8*7000 and (used>8*7000 if enabled else used==0)
                totals[leg]=dict(enabled=enabled,calls=n,fast=used,gil_acquires=n-used,acquire_ns=ns,max_ns=max_ns)
                for direction in ("forward","backward"):
                    assert entry["native_counters"][leg[-1]]["npu_embedding_"+direction+"_rebase_counter"]["accepted"]==8192
            (HERE/"audit.json").write_text(json.dumps(dict(lazy_init=totals,gate=entry),indent=2)+"\n")
        print("VERIFIED",pair,flush=True)


if __name__=="__main__":
    main()
