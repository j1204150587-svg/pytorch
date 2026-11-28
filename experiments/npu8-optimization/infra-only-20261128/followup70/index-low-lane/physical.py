"""Verify backward-stream submissions for each eager operator ordinal."""
from collections import Counter, defaultdict
import csv
import gzip
import json
from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
study = here.parents[1]
raw = Path(sys.argv[1])
out = here/"physical"
op = "aclnnEmbeddingDenseBackward"


def csvrows(path):
    with gzip.open(path,"rt",encoding="utf-8-sig",newline="") as fp:
        return list(csv.DictReader(fp))


def profile(leg):
    directory = out/leg
    if not (directory/"submissions.json").exists():
        subprocess.run([sys.executable,"-B",str(study/"queue/source-v2/analyze_dynamic_trace.py"),
                        str(raw/leg),str(directory)],check=True)
    calls = csvrows(directory/"queue_calls.csv.gz")
    work = csvrows(directory/"work_tasks.csv.gz")
    by_queue = defaultdict(list)
    for task in work:
        by_queue[task["step"],task["queue_id"]].append((task["stream"],task["task"]))
    result = {}
    for call in calls:
        if call["op"]!=op:
            continue
        key = (call["step"],call["role"],call["aten_ordinal_in_thread"])
        assert key not in result
        tasks = by_queue[call["step"],call["queue_id"]]
        assert len(tasks)==int(call["launch_api_count"])
        result[key] = (call,tasks)
    return result


a,b = map(profile,("01-A","02-B"))
assert a.keys()==b.keys()
report = {}
for step in ("4","5"):
    counts = Counter()
    for key in a:
        if key[0]!=step:
            continue
        ca, ta = a[key]
        cb, tb = b[key]
        assert ca["aten"]==cb["aten"] and ca["batch"]==cb["batch"] and ca["seq_len"]==cb["seq_len"]
        counts["requests"]+=1
        counts["workspace_A"]+=int(bool(int(ca["workspace_ns"] or 0)))
        counts["workspace_B"]+=int(bool(int(cb["workspace_ns"] or 0)))
        counts["ACL_launches_A"]+=len(ta)
        counts["ACL_launches_B"]+=len(tb)
        if ta==tb:
            counts["unchanged"]+=1
            continue
        deletions=[i for i,(_,name) in enumerate(ta) if "Cast" in name and ta[:i]+ta[i+1:]==tb]
        if deletions:
            counts["one_fewer_Cast_on_same_stream"]+=1
        else:
            counts["other_physical_changes"]+=1
    report[step]=dict(counts)
(out/"verified.json").write_text(json.dumps(report,indent=2))
for data in report.values():
    assert data["requests"]==1027 and data.get("one_fewer_Cast_on_same_stream")==1024, data
    assert not data.get("other_physical_changes") and data["workspace_B"]<=data["workspace_A"], data
