"""Measure the first and subsequent side-table requests in each observed training shape."""
from collections import Counter, defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
import statistics
import tarfile

here = Path(__file__).resolve().parent
def compressed_rows(archive,name):
    with gzip.open(archive.extractfile(name),"rt",encoding="utf-8-sig",newline="") as fp:
        return list(csv.DictReader(fp))
def u(row,key):
    return int(row[key] or 0)/1000
def quantile(values,pct):
    s=sorted(values)
    return s[min(len(s)-1,(len(s)*pct+99)//100-1)]
def summarize(group):
    keys=("aten_to_enqueue_ns","enqueue_ns","signed_queue_wait_ns","dequeue_ns","acl_api_sum_ns","workspace_ns")
    metrics={}
    for k in keys:
        vals=[u(r,k) for r in group if r[k]!=""]
        if vals:
            metrics[k.removesuffix("_ns")+"_us"]={"n":len(vals),"p50":statistics.median(vals),
                "p95":quantile(vals,95),"p99":quantile(vals,99),"max":max(vals)}
    return dict(requests=len(group),workspace_queries=sum(bool(r["workspace_ns"]) for r in group),
                launches=sum(int(r["launch_api_count"]) for r in group),latency=metrics)
report={}
with tarfile.open(here/"summary.tgz") as summary, tarfile.open(here/"raw.tgz") as raw:
    for arm,leg in (("A","01-A"),("B","02-B")):
        prefix="level1/"+arm+"/linked/"
        work=compressed_rows(summary,prefix+"work_tasks.csv.gz")
        calls=compressed_rows(summary,prefix+"queue_calls.csv.gz")
        name,=[m for m in raw.getnames() if m.startswith("raw/") and "-l1-" in m and "/"+leg+"/" in m and m.endswith("/kernel_details.csv")]
        actual=defaultdict(list)
        with io.TextIOWrapper(raw.extractfile(name),encoding="utf-8-sig",newline="") as fp:
            for row in csv.DictReader(fp):
                actual[row["Step Id"],row["Stream ID"],row["Task ID"],row["Name"]].append(row)
        owned=defaultdict(list)
        for task in work:
            owned[task["step"],task["queue_id"]].append(task)
        for step in ("4","5"):
            for op in ("aclnnEmbedding","aclnnEmbeddingDenseBackward"):
                group=[]
                kernels=Counter()
                for call in calls:
                    if call["op"]!=op or call["step"]!=step:
                        continue
                    tasks=owned[step,call["queue_id"]]
                    if not any("AsStrided" in t["task"] or "Slice" in t["task"] for t in tasks):
                        continue
                    assert len(tasks)==int(call["launch_api_count"]) and len({t["stream"] for t in tasks})==1
                    for t in tasks:
                        rows=actual[step,t["stream"],t["task_id"],t["task"]]
                        assert len(rows)==1,(step,t["stream"],t["task_id"],t["task"],len(rows))
                        k=rows[0]
                        kernels[k["Name"],k["Input Shapes"],k["Input Data Types"],k["Output Shapes"]]+=1
                    group.append(call)
                group.sort(key=lambda r:int(r["step_offset_ns"]))
                assert len(group)==1024 and len({r["role"] for r in group})==1
                report.setdefault(arm,{}).setdefault(step,{})[op]=dict(shape=[int(group[0]["batch"]),int(group[0]["seq_len"])],
                    first=summarize(group[:1]),subsequent=summarize(group[1:]),
                    physical=[dict(name=k[0],input_shapes=k[1],input_dtypes=k[2],output_shapes=k[3],executions=n)
                              for k,n in kernels.most_common()])
(here/"shape-heat.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
