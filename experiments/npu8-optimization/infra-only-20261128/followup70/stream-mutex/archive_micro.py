"""Verify the sampled NPU owner and preserve raw launch micro results and profiler exports."""
from datetime import datetime,timedelta
import json
from pathlib import Path
import sys
import tarfile


here=Path(__file__).resolve().parent
out=here/sys.argv[1]
root=here.parents[1]
read=lambda p:json.loads(p.read_bytes())
assert (out/"train.exit").read_text().strip()=="0"
known=read(out/"workers.json")
assert len(known)==4
a=datetime.fromisoformat(read(out/"before.card.json")["utc"])
b=datetime.fromisoformat(read(out/"after.card.json")["utc"])+timedelta(seconds=1)
ids={r["host_pid"] for r in known.values()}
covered=set()
samples=[]
for line in (root/"followup70-20261128/chip.jsonl").read_bytes().splitlines():
    row=json.loads(line)
    if a<=datetime.fromisoformat(row["utc"])<b:
        samples.append(row)
        assert row["valid"]
        for proc in row["processes"]:
            assert proc["owner"]=="training" and proc["pid"] in ids,(proc,ids)
            covered.add(proc["pid"])
assert covered,known
slot=dict(workers=known,snapshots=len(samples),observed_in_chip_samples=sorted(covered),
          missed_by_chip_cadence=sorted(ids-covered),external=[])
(out/"micro-slot.json").write_text(json.dumps(slot,indent=2)+"\n")
(out/"chip.jsonl").write_text("".join(json.dumps(r)+"\n" for r in samples))
with tarfile.open(here/"micro-evidence.tar.gz","w:gz",compresslevel=2) as arc:
    for f in sorted(out.rglob("*")):
        if not f.is_file(): continue
        rel=f.relative_to(out)
        if len(rel.parts)==1 or f.name in ("trace_view.json","kernel_details.csv","step_trace_time.csv","op_statistic.csv"):
            arc.add(f,arcname=rel.as_posix())
