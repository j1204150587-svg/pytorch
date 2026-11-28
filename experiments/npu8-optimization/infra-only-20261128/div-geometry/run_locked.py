"""Gate the candidate and launch each authorized pair through the archived runner and its host locks."""
import json
from pathlib import Path
import re
import subprocess
import sys

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
INNER="/artifacts/"+STUDY.name
NAME="div-geometry"


def read(p):
    return json.loads(p.read_bytes())


def call(cmd,log):
    with log.open("wb") as out:
        subprocess.run([str(c) for c in cmd],stdout=out,stderr=subprocess.STDOUT,check=True)


def main():
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage",choices=["micro","audit","formal"])
    args=ap.parse_args()
    receipt=read(HERE/"build/build-receipt.json")
    if receipt["status"]!="built":
        raise ValueError("Build incomplete")
    paths=STUDY/"recipes"
    if args.stage=="micro":
        call([sys.executable, STUDY/"make_recipe.py",NAME,receipt["library"],receipt["library_sha256"],"NPU_DIV_GEOMETRY_KEY"],
             HERE/"make-recipe.log")
        for audit in (False,True):
            p=paths/(NAME+("-audit" if audit else "")+".json")
            r=read(p)
            r["common"].update(NPU_DYNAMIC_REBASE_INDICES="1",NPU_DYNAMIC_REBASE_FORWARD="1",
                                NPU_DYNAMIC_REBASE_COUNT="0",ACLNN_CACHE_LIMIT="65536",
                                NPU_DIV_GEOMETRY_COUNT=str(int(audit)),NPU_DIV_GEOMETRY_AUDIT="0")
            p.write_text(json.dumps(r,indent=2)+"\n")
        # Take the admission lock first, then let operator_probe take the same physical reservation as the stock runner.
        call(["flock","-x","/run/lock/recsys-gr-ds019-npu8-infra-study.lock",sys.executable, STUDY/"operator_probe.py",
              "--package",INNER+"/stock-overlay/"+NAME,HERE/"micro-run-v2",INNER+"/"+NAME+"/micro.py",
              "--schedule","@WORK/configs/dynamic_shapes_npu8.json","--output","@OUT/micro.json"],HERE/"micro.log")
        return
    micro=read(HERE/"micro-run-v2/micro.json")
    if not (micro["complete"] and all(c["passed"] for c in micro["cases"])):
        raise ValueError("Native micro gate did not pass")
    phases=[("audit","ab")] if args.stage=="audit" else [("no-profiler","ab"),("no-profiler","ba"),("l0","ab"),("l0","ba"),("l1","ab")]
    for stage,order in phases:
        out=STUDY/"pairs"/(NAME+"-"+stage+"-"+order)
        if out.exists():
            r=read(out/"result.json")
            if r["exit"] or (stage=="audit" and not r.get("audit_passed")):
                raise ValueError(out)
        else:
            if stage!="audit" and not (HERE/"audit-summary.json").exists():
                raise ValueError("Missing training gate")
            call([sys.executable,STUDY/"runner/host_pair.py","--runner",STUDY/"archived-runner",
                  "--recipe",paths/(NAME+("-audit" if stage=="audit" else "")+".json"),
                  "--out",out,"--mode",stage,"--order",order],HERE/(stage+"-"+order+".log"))
            r=read(out/"result.json")
        if stage=="audit":
            log=(Path(r["directory"])/"02-B/run.log").read_text()
            m=re.search(r"DIV_GEOMETRY fallback=(\d+) first=(\d+) reuse=(\d+) checked=(\d+) exec_hit=(\d+) exec_miss=(\d+)",log)
            if not m:
                raise ValueError("No native activation report")
            fb,first,reuse,checked,hits,misses=map(int,m.groups())
            if not (r.get("audit_passed") and first+reuse==8192 and reuse>8000 and hits+misses==8192 and checked==0):
                raise ValueError((r.get("audit_passed"),m.group(0)))
            (HERE/"audit-summary.json").write_text(json.dumps({"native_counts":m.group(0),"result":r},indent=2))


if __name__=="__main__":
    main()
