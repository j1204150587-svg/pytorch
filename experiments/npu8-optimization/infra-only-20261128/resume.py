"""Resume independent native pairs when logical NPU8 is free. Preserve interrupted attempts."""
import fcntl
import json
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time
import verify_slot

ROOT=Path(__file__).resolve().parent

def log(msg):
    print(datetime.now().astimezone().isoformat(timespec="seconds"), msg, flush=True)

def available():
    query=subprocess.run(["npu-smi","info","-t","proc-mem","-i","4","-c","0"],
                         text=True,capture_output=True,timeout=20)
    return query.returncode==0 and "No process in device" in query.stdout

def retire(p,reason):
    new=ROOT/"excluded"/(p.name+"-"+str(time.time_ns()))
    new.parent.mkdir(exist_ok=True)
    p.rename(new)
    (new/"excluded-reason.txt").write_text(reason+"\n")
    log(f"preserved {new.name}: {reason}")

def verified(pair):
    result=pair/"result.json"
    if not result.is_file(): raise RuntimeError("unfinished pair "+str(pair))
    r=json.loads(result.read_bytes())
    if r["exit"]:
        status=json.loads((Path(r["directory"])/"status.json").read_bytes()) if r.get("directory") else {}
        detail=str(status.get("detail",""))
        if r["exit"] in (10,12) and ("occupied" in detail or "host lock" in detail or "pid" in detail.lower()):
            retire(pair,detail)
            return False
        raise RuntimeError(str(r)+" "+detail)
    sampled=[(datetime.fromisoformat(x["utc"]),x) for line in (ROOT/"chip.jsonl").read_bytes().splitlines() if line
             for x in [json.loads(line)]]
    report=verify_slot.run(ROOT,sampled,None)[pair.name]
    if any(x["unknown_or_external"] or x["invalid"] or not x["snapshots_with_worker"] for x in report):
        retire(pair,"chip samples incomplete or show a competing worker: "+json.dumps(report))
        return False
    return True

def main():
    # Successful audits have already run; each performance recipe belongs to that candidate.
    wanted=[(n,mode,order) for n in ("stock","addmm","divs","queue","emb-back","emb-fwd","mm")
            for mode,order in ([("no-profiler","ab"),("l0","ab")] if n=="stock" else
            [("no-profiler","ab"),("no-profiler","ba"),("l0","ab"),("l0","ba"),("l1","ab")])]
    for name,mode,order in wanted:
        pair=ROOT/"pairs"/f"{name}-{mode}-{order}"
        if name not in ("stock",):
            gate=ROOT/"pairs"/("addmm-audit1" if name=="addmm" else name+"-audit-ab")/"result.json"
            assert json.loads(gate.read_bytes())["audit_passed"]
        while not pair.is_dir() or not verified(pair):
            waiting=False
            while not available():
                if not waiting:
                    log("waiting for NPU8 before "+pair.name)
                    waiting=True
                time.sleep(20)
            log("starting "+pair.name)
            cmd=[sys.executable,"-B",str(ROOT/"runner/host_pair.py"),"--runner",str(ROOT/"archived-runner"),
                 "--recipe",str(ROOT/"recipes"/(name+".json")),"--out",str(pair),"--mode",mode,"--order",order]
            subprocess.run(cmd,cwd=ROOT,check=False)
        log("verified "+pair.name)
    log("complete")
if __name__=="__main__":
    with (ROOT/"recovery.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        main()

