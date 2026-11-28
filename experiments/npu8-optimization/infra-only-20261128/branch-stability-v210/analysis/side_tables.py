"""Verify first and following side-table requests by eager ordinal and physical work, from Level1."""
import argparse
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
import csv
from decimal import Decimal
import gzip
import json
import math
from pathlib import Path
import statistics

from measure import digest, json_text, linked_csv, local_path, write


SIDE = {
    "forward": ("aclnnEmbedding", "aten::embedding", "main", range(3, 1027)),
    "backward": ("aclnnEmbeddingDenseBackward", "aten::embedding_dense_backward", "worker", range(1024)),
}
STEPS = ("4", "5")


def distribution(values):
    values = sorted(v / 1000 for v in values)
    if not values:
        return {"n": 0}
    return {
        "n": len(values), "p50": statistics.median(values),
        "p95": values[math.ceil(0.95 * len(values)) - 1],
        "p99": values[math.ceil(0.99 * len(values)) - 1], "max": values[-1],
    }


def nanoseconds(value):
    return int(Decimal(str(value)) * 1000)


def intervals(trace, queue_ids):
    with gzip.open(trace, "rt", encoding="utf-8-sig") as src:
        data = json.load(src)
    events = data if isinstance(data, list) else data["traceEvents"]
    queues = {"enqueue": {}, "dequeue": {}}
    owners = {}
    bands = defaultdict(list)
    aten_names = {definition[1] for definition in SIDE.values()}
    ws_names = {"AscendCL@" + definition[0] + "GetWorkspaceSize" for definition in SIDE.values()}
    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        name = event["name"]
        if category in queues:
            key = str(event["args"]["correlation_id"])
            if key not in queue_ids:
                continue
            start = nanoseconds(event["ts"])
            queues[category][key] = (event["tid"], start, start + nanoseconds(event["dur"]))
        elif name in aten_names:
            start = nanoseconds(event["ts"])
            owners[(event["tid"], start, name)] = start + nanoseconds(event["dur"])
        elif name in ws_names or name.startswith(("AscendCL@aclrtLaunchKernel", "AscendCL@aclrtMemcpy")):
            start = nanoseconds(event["ts"])
            family = name if name in ws_names else "acl"
            bands[event["tid"], family].append((start, start + nanoseconds(event["dur"]), name))
    indexed = {}
    for key, spans in bands.items():
        spans.sort()
        indexed[key] = ([e[0] for e in spans], spans)
    for lane in queues.values():
        assert set(lane) == queue_ids
    return queues, owners, indexed


def contained(index, tid, family, lo, hi):
    starts, spans = index.get((tid, family), ([], []))
    return [span for span in spans[bisect_left(starts, lo):bisect_right(starts, hi)] if span[1] <= hi]


def counter_rows(counter, fields):
    return [dict(zip(fields, key), n=n) for key, n in sorted(counter.items(), key=lambda item: str(item[0]))]


def summarize(records):
    durations = {
        "producer": [int(r["call"]["aten_to_enqueue_ns"]) for r in records],
        "workspace": [duration for r in records for duration in r["workspace"]],
        "enqueue": [int(r["call"]["enqueue_ns"]) for r in records],
        "signed_queue_wait": [int(r["call"]["signed_queue_wait_ns"]) for r in records],
        "consumer": [int(r["call"]["dequeue_ns"]) for r in records],
        "acl_single": [duration for r in records for duration in r["acl"]],
        "acl_sum_per_request": [sum(r["acl"]) for r in records],
    }
    geometry = Counter(task for r in records for task in r["geometry"])
    api_names = Counter(name for r in records for name in r["api_names"])
    return {
        "requests": len(records), "workspace_queries": len(durations["workspace"]),
        "requests_with_workspace": sum(bool(r["workspace"]) for r in records),
        "acl_submissions": len(durations["acl_single"]), "physical_tasks": sum(geometry.values()),
        "acl_api_counts": dict(sorted(api_names.items())),
        "latency_us": {field: distribution(samples) for field, samples in durations.items()},
        "gap_ms": {field: sum(int(g[field] or 0) for r in records for g in r["gaps"]) / 1e6
                   for field in ("gap_ns", "workspace_ns", "aten_to_enqueue_remainder_ns", "visible_queue_wait_ns",
                                 "consumer_before_api_ns", "acl_submission_api_ns", "after_api_ns")},
        "kernel_inputs": counter_rows(geometry, ("stream", "kernel", "task_type", "input_shapes", "input_dtypes")),
    }


def read_leg(linked, raw):
    trace = raw / "trace_view.json.gz"
    kernel_file, = raw.rglob("kernel_details.csv")
    queue_file, work_file, gap_file = [linked/(stem+".csv.gz") for stem in ("queue_calls", "work_tasks", "device_gaps")]
    calls = linked_csv(queue_file)
    chosen = {}
    non_side = {}
    for direction, (operation, aten, role, selected) in SIDE.items():
        for step in STEPS:
            all_rows = [r for r in calls if r["step"] == step and r["op"] == operation]
            assert len(all_rows) == 1027
            assert {r["aten"] for r in all_rows} == {aten} and {r["role"] for r in all_rows} == {role}
            assert len({r["producer_tid"] for r in all_rows}) == 1
            ordered = sorted(all_rows, key=lambda r: int(r["step_offset_ns"]))
            assert [int(r["aten_ordinal_in_thread"]) for r in ordered] == list(range(1027))
            chosen[step, direction] = {o: ordered[o] for o in selected}
            non_side[step,direction] = [r["queue_id"] for o,r in enumerate(ordered) if o not in selected]
    queue_ids = {row["queue_id"] for selection in chosen.values() for row in selection.values()}
    assert len(queue_ids) == 4096
    work = defaultdict(list)
    marker_ids = set()
    for task in linked_csv(work_file):
        if "AsStrided" in task["task"] or "Slice" in task["task"]:
            marker_ids.add(task["queue_id"])
        if task["queue_id"] in queue_ids:
            work[task["queue_id"]].append(task)
    gaps = defaultdict(list)
    for gap in linked_csv(gap_file):
        if gap["state"] == "linked" and gap["queue_id"] in queue_ids:
            gaps[gap["queue_id"]].append(gap)
    physical_keys = {(t["step"],t["stream"],t["task_id"],t["task"]) for tasks in work.values() for t in tasks}
    geometry = {}
    with kernel_file.open(encoding="utf-8-sig", newline="") as src:
        for kernel in csv.DictReader(src):
            key = (kernel["Step Id"],kernel["Stream ID"],kernel["Task ID"],kernel["Name"])
            if key in physical_keys:
                assert key not in geometry
                geometry[key] = (kernel["Input Shapes"],kernel["Input Data Types"])
    assert set(geometry) == physical_keys
    queues, owners, api_index = intervals(trace, queue_ids)
    signatures, summaries = {}, {}
    for (step, direction), selection in chosen.items():
        ordered = []
        for ordinal, call in selection.items():
            key = call["queue_id"]
            tid, enq_start, _ = queues["enqueue"][key]
            consumer_tid, deq_start, deq_end = queues["dequeue"][key]
            start = enq_start-int(call["aten_to_enqueue_ns"])
            assert owners[tid, start, call["aten"]] >= enq_start
            ws_name = "AscendCL@" + call["op"] + "GetWorkspaceSize"
            ws_spans = contained(api_index, tid, ws_name, start, enq_start)
            workspace = [end-begin for begin,end,_ in ws_spans]
            assert len(workspace) <= 1 and sum(workspace)==int(call["workspace_ns"] or 0)
            launches=contained(api_index, consumer_tid, "acl", deq_start, deq_end)
            names=[name for _,_,name in launches]
            durations=[end-begin for begin,end,_ in launches]
            assert "|".join(names)==call["acl_api_sequence"] and sum(durations)==int(call["acl_api_sum_ns"])
            jobs = sorted(work[key],key=lambda t:int(t["stream_work_slot"]))
            assert len(launches)==int(call["launch_api_count"]) and jobs
            assert len({t["stream"] for t in jobs})==1
            actual = [(t["stream"],t["task"],t["task_type"],*geometry[step,t["stream"],t["task_id"],t["task"]]) for t in jobs]
            signature = tuple((t["stream"],t["task"],t["task_type"]) for t in jobs)
            slots = tuple(int(t["stream_work_slot"]) for t in jobs)
            signatures[step,direction,ordinal] = (signature,slots,tuple(names))
            ordered.append(dict(call=call,workspace=workspace,acl=durations,api_names=names,
                                geometry=actual,gaps=gaps[key]))
        first = ordered[0]["call"]
        shape = [int(first["batch"]), int(first["seq_len"])]
        assert all([int(row["call"]["batch"]),int(row["call"]["seq_len"])]==shape for row in ordered)
        lookup_ranges=[]
        for position,row in enumerate(ordered,1):
            if not row["workspace"]:
                continue
            if lookup_ranges and lookup_ranges[-1][1]+1==position:
                lookup_ranges[-1][1]=position
            else:
                lookup_ranges.append([position,position])
        summaries.setdefault(step,{})[direction] = {
            "step_shape": shape, "first_seen_step": int(first["first_step_with_input_shape"]),
            "workspace_side_table_positions_1based": lookup_ranges,
            "slice_or_as_strided_work_marker": {
                "selected": sum(c["queue_id"] in marker_ids for c in selection.values()),
                "other_embedding_requests": sum(k in marker_ids for k in non_side[step,direction]),
            },
            "first": summarize(ordered[:1]), "following": summarize(ordered[1:]),
        }
    files = {f.name: digest(f) for f in (trace,kernel_file,queue_file,work_file,gap_file)}
    return summaries,signatures,files


def paired_ordinals(a,b):
    assert a.keys()==b.keys()
    output={}
    for step in STEPS:
        for direction, (_,_,_,indices) in SIDE.items():
            cohorts = {}
            for name, ordinals in (("first",indices[:1]),("following",indices[1:])):
                patterns = Counter()
                examples = {}
                same_slots = same_streams = same_count = same_sequence = same_acl = 0
                for ordinal in ordinals:
                    ast,asl,an = a[step,direction,ordinal]
                    bst,bsl,bn = b[step,direction,ordinal]
                    streams_a = tuple(dict.fromkeys(job[0] for job in ast))
                    streams_b = tuple(dict.fromkeys(job[0] for job in bst))
                    same_streams += streams_a==streams_b
                    same_count += len(ast)==len(bst)
                    same_slots += asl==bsl and streams_a==streams_b
                    same_sequence += ast==bst
                    same_acl += an==bn
                    key=(ast,bst)
                    patterns[key]+=1
                    examples.setdefault(key,ordinal)
                pattern_rows=[dict(A=ak,B=bk,n=count,example_ordinal=examples[ak,bk])
                              for (ak,bk),count in patterns.most_common()]
                cohorts[name] = dict(checked=len(ordinals),ordinal_min=ordinals[0],ordinal_max=ordinals[-1],
                    same_stream=same_streams, same_work_count=same_count, same_per_stream_slots=same_slots,
                    same_stream_and_work_sequence=same_sequence, same_acl_sequence=same_acl,
                    physical_patterns=pattern_rows)
            output.setdefault(step,{})[direction]=cohorts
    return output


def run(root, comparisons):
    result={}
    for name in comparisons:
        folder = root/"reports"/name
        metrics=json.loads((folder/"metrics.json").read_bytes())
        row,=[p for p in metrics["pairs"] if p["mode"]=="l1"]
        detail=folder/row["level1_file"]
        raw=local_path(root/"data/manifests",row["raw"])
        schedule=json.loads((root/"data/dynamic_shapes_npu8.json").read_bytes())
        comparison={"pair":row["id"],"order":row["order"],"branch":metrics["branch_commit"],
                    "pair_file":str(detail.relative_to(root).as_posix()),"inputs":{}}
        signs={}
        for leg_order,leg in enumerate(row["order"].upper(),1):
            env=row["proof"]["legs"][leg]["worker_env"]
            enabled=str(int(leg=="B"))
            assert all(env[k]==enabled for k in ("NPU_DYNAMIC_REBASE_INDICES","NPU_DYNAMIC_REBASE_FORWARD","TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY"))
            assert env.get("ACLNN_CACHE_LIMIT")==("65536" if name=="fixed-cache" else None)
            linked=local_path(detail.parent,leg+"/linked")
            summaries,signs[leg],inputs=read_leg(linked,raw/f"{leg_order:02d}-{leg}")
            for step, directions in summaries.items():
                for record in directions.values():
                    assert record["step_shape"]==schedule[int(step)] and record["first_seen_step"]==schedule.index(record["step_shape"])
            comparison[leg]=summaries
            comparison["inputs"][leg]=inputs
        comparison["paired_ordinals"]=paired_ordinals(signs["A"],signs["B"])
        result[name]=comparison
    return result

def cells(a, b, field, percentile=None):
    if percentile:
        return f"{a['latency_us'][field][percentile]:.2f} → {b['latency_us'][field][percentile]:.2f}"
    return f"{a[field]:,} → {b[field]:,}"


def markdown(results):
    lines=["# 动态 eager 侧表 embedding 的首次与后续提交","",
        "同一目标 wheel 的 A 腿关闭三个改动，B 腿全部开启。每种缓存各一对正序 Level1，使用真实第 4 步 [1,19] 和第 5 步 [2,14]，两种输入形状均在对应步骤首次出现。每次前向或反向有 1,027 个同名请求。按同一生产线程、ATen 算子及该步的同名调用序号，前向的 3–1026、反向的 0–1023 对应 1,024 张侧表。首张指该方向当步第一个访问的侧表。反向按自己的访问顺序计数。后续 1,023 次使用同一步的输入形状，无法据此判断某个 executor 键过去是否见过。","",
        "GetWorkspaceSize 计数逐个匹配 ATen 进入至 enqueue 开始间的同名查询。producer 从对应 ATen 进入计到 enqueue 开始，consumer 为完整 dequeue，单 ACL 值是每一次 launch 或 memcpy API 的时长。以下 A → B 比较单位为 µs。首张的 producer、consumer 是单请求值；其单 ACL p95 在该请求内的实际提交上计算。",""]
    for group,title in (("first","首张侧表"),("following","后续 1,023 张侧表")):
        lines.extend(["## "+title,"","| 缓存上限 | 步 / 形状 | 方向 | workspace 查询 A → B | producer p95 | 有符号队列等待 p95 | consumer p95 | 单 ACL p95 |",
                      "|---|---|---|---:|---:|---:|---:|---:|"])
        for name,pair in results.items():
            label="65,536" if name=="fixed-cache" else "默认 10,000"
            for step in STEPS:
                for direction in SIDE:
                    a=pair["A"][step][direction][group]
                    b=pair["B"][step][direction][group]
                    shape=pair["A"][step][direction]["step_shape"]
                    text="前向" if direction=="forward" else "反向"
                    columns=[label,f"{step} / {shape}",text,cells(a,b,"workspace_queries")]
                    columns.extend(cells(a,b,field,"p95") for field in ("producer","signed_queue_wait","consumer","acl_single"))
                    lines.append("| "+" | ".join(columns)+" |")
        lines.append("")
    same_locations=defaultdict(list)
    for name,pair in results.items():
        label="65,536" if name=="fixed-cache" else "默认 10,000"
        layouts={tuple(tuple(span) for span in pair["B"][step][direction]["workspace_side_table_positions_1based"])
                 for step in STEPS for direction in SIDE}
        if len(layouts)==1:
            same_locations[next(iter(layouts))].append(label)
    for spans,labels in same_locations.items():
        locations="、".join(f"{a}–{b}" if a!=b else str(a) for a,b in spans)
        lines.extend([f"在{' 和 '.join(labels)} 的 B 腿，两步前向与反向的 workspace 查询均落在各自访问顺序的第 {locations} 张侧表。",""])
    lines.extend(["## 相同 eager ordinal 的物理流与底层任务","",
        "每行 1,024 个请求。表中按 ATen 序号配对，不跨进程比较 task ID。流槽位使用每条物理流上的工作任务顺序（stream_work_slot）。任务序列同时比较流、任务名与类型。每个实际硬件任务另用 kernel_details.csv 的步骤、流、task ID 和算子名唯一核对，输入形状和 dtype 分布见 JSON。","",
        "| 缓存上限 | 步 | 方向 | 实际 ACL 提交 A → B | 物理任务 A → B | 物理流 A → B | 同任务数 / 同任务序列 / 同流槽位的请求数 |",
        "|---|---:|---|---:|---:|---|---:|"])
    transformations=defaultdict(list)
    for name,pair in results.items():
        label="65,536" if name=="fixed-cache" else "默认 10,000"
        for step in STEPS:
            for direction in SIDE:
                text="前向" if direction=="forward" else "反向"
                match=pair["paired_ordinals"][step][direction]
                a=pair["A"][step][direction]
                b=pair["B"][step][direction]
                groups=("first","following")
                def totals(record,field):
                    return sum(record[group][field] for group in groups)
                def streams(record):
                    return ",".join(sorted({k["stream"] for group in groups for k in record[group]["kernel_inputs"]}))
                paired=[sum(match[group][k] for group in groups) for k in ("same_work_count","same_stream_and_work_sequence","same_per_stream_slots")]
                acl=f"{totals(a,'acl_submissions'):,} → {totals(b,'acl_submissions'):,}"
                physical=f"{totals(a,'physical_tasks'):,} → {totals(b,'physical_tasks'):,}"
                lines.append(f"| {label} | {step} | {text} | {acl} | {physical} | {streams(a)} → {streams(b)} | "+" / ".join(map(str,paired))+" |")
                changed=Counter()
                for group in groups:
                    for row in match[group]["physical_patterns"]:
                        if row["A"]!=row["B"]:
                            a_tasks=" → ".join(t[1].split("_")[-1] for t in row["A"])
                            b_tasks=" → ".join(t[1].split("_")[-1] for t in row["B"])
                            changed[a_tasks,b_tasks]+=row["n"]
                for (old,new),count in changed.most_common():
                    transformations[text,old,new,count].append((label,step))
    if transformations:
        lines.append("")
    for (direction,old,new,count),conditions in transformations.items():
        scope="两种容量的两个步骤" if len(conditions)==len(results)*len(STEPS) else "、".join(f"{label} 第 {step} 步" for label,step in conditions)
        lines.append(f"- {scope}的{direction}各有 {count:,} 个同序号请求从 `{old}` 变为 `{new}`。")
    lines.extend(["","P50 取中位数，p95/p99 取升序最近秩，保留最大值。完整 JSON 按首张/后续及四个真实提交阶段记录分位、实际底层序列、同序号配对计数、硬件输入和关联空档。单个 ACL 调用与每个请求内累计 ACL 时间分列。Level1 的延迟不直接换算无 profiler 步时或 Level0 Free。[复算数据](side-tables.json) · [脚本](../analysis/side_tables.py)",""])
    return "\n".join(lines)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args=cli.parse_args()
    records=run(args.root,("fixed-cache","default-cache"))
    write(args.root/"reports/side-tables.json",json_text({"comparison":records}))
    dest=args.root/"reports/side-tables.md"
    write(dest,markdown(records))
    print(dest)
