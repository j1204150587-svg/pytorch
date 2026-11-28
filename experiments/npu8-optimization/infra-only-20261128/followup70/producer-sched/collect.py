"""Collect profiler outputs and matching on-host sidecars under the same rank receipts."""
from datetime import datetime,timedelta
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import time

from sample import CAP,HERE
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from verify_slot import run as verify
from export import add_raw

def read(p):return json.loads(p.read_bytes())
def main():
    deadline=time.monotonic()+3600
    while time.monotonic()<deadline:
        if (HERE/"pair/result.json").exists() and all((CAP/l/"anchor.txt").exists() for l in ("01-A","02-B")):break
        time.sleep(3)
    else: raise TimeoutError("pair or host sidecars")
    time.sleep(3)
    pair=HERE/"pair"
    result=read(pair/"result.json")
    assert result["exit"]==0
    raw=Path(result["directory"])
    all_bounds=[]
    for leg in ("01-A","02-B"):
        w=read(pair/leg/"worker.json")
        assert w["pid"]==read(CAP/leg/"receipt.json")["pid"]
        out=CAP/leg/"enriched"
        out.mkdir(exist_ok=True)
        traces=list((raw/leg).rglob("trace_view.json.gz"))
        assert len(traces)==1,traces
        trace,=traces
        with tempfile.TemporaryDirectory(dir=CAP) as td:
            t=Path(td)
            (t/"trace_view.json.gz").symlink_to(trace)
            (t/"rank000").symlink_to(raw/leg/"rank000",target_is_directory=True)
            subprocess.run([sys.executable,"-B",str(HERE/"analyze_dynamic_trace.py"),str(t),str(out)],check=True)
        subprocess.run([sys.executable,"-B",str(HERE/"clock_bounds.py"),str(trace),str(HERE/"analyze_dynamic_trace.py"),str(out/"device-clock.json")],check=True)
        subprocess.run([sys.executable,"-B",str(HERE/"resolve.py"),str(CAP/leg),str(Path("/proc")/str(w["init_pid"])/"root")],check=True)
        subprocess.run([sys.executable,"-B",str(HERE/"analyze.py"),str(trace),str(out),str(CAP/leg)],check=True)
        (HERE/(leg+"-sched.json")).write_bytes((out/"sched-analysis.json").read_bytes())
        all_bounds.append((datetime.fromisoformat(read(raw/leg/"before.card.json")["utc"]),datetime.fromisoformat(read(raw/leg/"after.card.json")["utc"])+timedelta(seconds=1)))
    snapshots=[]
    for ln in (ROOT/"followup70-20261128/chip.jsonl").read_bytes().splitlines():
        v=json.loads(ln)
        t=datetime.fromisoformat(v["utc"])
        if any(a<=t<b for a,b in all_bounds):snapshots.append((t,v))
    with tempfile.TemporaryDirectory(dir=CAP) as tmp:
        t=Path(tmp);(t/"pairs").mkdir();(t/"pairs/producer-sched-aa").symlink_to(pair,target_is_directory=True)
        slot=verify(t,snapshots,None)
    assert all(not v["invalid"] and not v["unknown_or_external"] and v["snapshots_with_worker"] for arms in slot.values() for v in arms),slot
    (HERE/"slot.json").write_text(json.dumps(slot,indent=2)+"\n")
    (CAP/"chip.jsonl").write_text("".join(json.dumps(v)+"\n" for t,v in snapshots))
    with tarfile.open(CAP/"raw-evidence.tar.gz","w:gz",compresslevel=2) as arc:
        arc.add(pair,arcname="pair");add_raw(arc,raw)
        arc.add(ROOT/"archived-runner/dynamic_shapes_npu8.json",arcname="schedule.json")
        for n in ("chip.jsonl","01-A","02-B"):arc.add(CAP/n,arcname=n)
        for n in ("slot.json","pair.log","sample.log","sampler-initial.log","format.log","format.txt","recipe.json"):
            arc.add(HERE/n,arcname=n)
    print("complete",CAP,flush=True)

if __name__=="__main__":main()

