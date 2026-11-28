"""Bound driver return timing using per-task submit starts and global device drains."""
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sys

out = Path(sys.argv[1])
acls = json.loads((out/"clock-bounds.json").read_bytes())
with gzip.open(out/"calls.csv.gz","rt",encoding="utf-8-sig",newline="") as fp:
    calls = list(csv.DictReader(fp))
report = {}
for step in ("4","5"):
    group = [r for r in calls if r["step"]==step and r.get("device_start_ns")]
    multiplicity = Counter((r["tid"],r["acl_start_ns"]) for r in group)
    assert max(multiplicity.values())==1
    clock = acls["ProfilerStep#"+step]
    lower = max(clock["clock_lower_ns"],max(int(r["start_ns"])-int(r["device_start_ns"]) for r in group))
    upper = clock["clock_upper_ns"]
    assert lower<=upper
    gaps = [r for r in group if r.get("gap_left_ns")]
    def after(offset, subset):
        return sum(max(0,int(r["gap_right_ns"])+offset-max(int(r["gap_left_ns"])+offset,int(r["end_ns"])))
                   for r in subset)/1e6
    def measure(subset):
        return dict(count=len(subset), gap_ms=sum(int(r["gap_ns"]) for r in subset)/1e6,
                    hal_inside_nominal_ms=sum(int(r["hal_in_gap_ns"]) for r in subset)/1e6,
                    after_hal_ms=dict(earliest=after(lower,subset),nominal=after(0,subset),latest=after(upper,subset)))
    report[step]=dict(shape=json.loads(gaps[0]["shape"]), lower_ns=lower,upper_ns=upper,
                      checked_physical=len(group),all=measure(gaps),
                      by_stream={s:measure([r for r in gaps if r["stream"]==s])
                                 for s in sorted({r["stream"] for r in gaps})})
(out/"hal-bounds.json").write_text(json.dumps(report,indent=2))
