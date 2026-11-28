"""Pair dynamic Level1 submissions, physical stream slots and Host/gap intervals."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


OPERATORS = {
    "forward": ("aclnnEmbedding", "aten::embedding", "main"),
    "backward": ("aclnnEmbeddingDenseBackward", "aten::embedding_dense_backward", "worker"),
}
LATENCY = ("aten_to_enqueue_ns", "signed_queue_wait_ns", "dequeue_ns", "aclrtLaunchKernelWithHostArgs")
GAPS = ("gap_ns", "workspace_ns", "visible_queue_wait_ns", "consumer_before_api_ns",
        "acl_submission_api_ns", "after_api_ns")


def read(path):
    return json.loads(path.read_bytes())


def csv_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def compare(raw, data, tools):
    def run(name, *args):
        subprocess.run([sys.executable, "-B", str(tools / name), *map(str, args)], check=True)
    run("compare_dynamic_legs.py", raw / "01-A", raw / "02-B", data)
    for direction, (op, aten, _) in OPERATORS.items():
        run("compare_operator_calls.py", data, "--aten", aten, "--op", op, "--out", data / direction)
    run("stream_substitutions.py", data)
    for leg in ("A", "B"):
        run("first_visible.py", data / leg)


def collect(raw, data):
    latency = {(int(r["step"]), r["op"], r["aten"], r["role"], r["metric"]): r
               for r in csv_rows(data / "operator_latency.csv")}
    gaps = {(int(r["step"]), r["op"], r["role"], r["metric"]): r
            for r in csv_rows(data / "gap_attribution.csv")}
    summarized = {"original_run": raw.name, "operators": {}, "full_step": {}}
    for name, (op, aten, role) in OPERATORS.items():
        info = summarized["operators"][name] = {}
        for step in (4, 5):
            timing = {k: latency[(step, op, aten, role, k)] for k in (*LATENCY, "workspace_ns")}
            intervals = {k: gaps[(step, op, role, k)] for k in GAPS}
            counts, = [r for r in csv_rows(data / name / "counts.csv") if int(r["step"]) == step and r["role"] == role]
            info[step] = {
                "shape": [int(timing[LATENCY[0]][s]) for s in ("batch", "seq_len")],
                "matched_physical_stream_slots": {k: int(v) for k, v in counts.items() if k not in ("step", "role")},
            }
            for leg in ("A", "B"):
                def val(row, column):
                    value = row.get(leg + "_" + column)
                    return float(value) if value else None
                info[step][leg] = {
                    "requests": int(val(timing[LATENCY[0]], "n")),
                    "kernel_launches": int(val(timing[LATENCY[-1]], "n")),
                    "workspace_calls": int(val(timing["workspace_ns"], "n") or 0),
                    "workspace_ms": val(timing["workspace_ns"], "sum_ms") or 0,
                    "latency_us": {key: {q: val(timing[key], q + "_us") for q in ("p50", "p95", "p99", "max")}
                                   for key in LATENCY},
                    "gap_ms": {key: val(row, "sum_ms") or 0 for key, row in intervals.items()},
                    "gaps_n": int(val(intervals["gap_ns"], "n")),
                    "onset_of_successor_gap": {r["first_visible"]: int(r["n"]) for r in
                                                csv_rows(data / leg / "first_visible.csv")
                                                if int(r["step"]) == step and (r["op"], r["role"]) == (op, role)},
                }
    global_sub = read(data / "submissions_compare.json")["steps"]
    substitutions = read(data / "physical_stream_substitutions.json")
    for step in (4, 5):
        comparison, = (r for r in global_sub if r["step"] == step)
        entry = summarized["full_step"][step] = {"comparison": comparison,
                                                 "physical_streams": [r for r in substitutions if r["step"] == step]}
        for leg in ("A", "B"):
            r, = (x for x in read(data / leg / "submissions.json")["steps"] if x["step"] == step)
            entry[leg] = {k: r[k] for k in ("acl_submission_apis", "hardware_types", "first_step_with_input_shape")}
            profiled, = (x for x in csv_rows(raw / {"A":"01-A", "B":"02-B"}[leg] / "step_trace_time.csv")
                         if int(x["Step"]) == step)
            entry[leg]["level1_ms"] = {k: float(profiled[k])/1000 for k in ("Free", "Stage", "Computing")}
    (data / "embedding_metrics.json").write_text(json.dumps(summarized, indent=2) + "\n")
    return summarized


def note(result, data, target):
    def numbers(a, b, digits=2):
        return f"{a:.{digits}f} → {b:.{digits}f}"
    out = [f"# {target} 动态训练 Level1 归因", "",
           f"原始 A/B 会话 `{result['original_run']}`。#4 为 [1,19]，#5 为 [2,14]。"
           "两步的整步输入均为本次训练中首次出现的 shape。", "",
           "| 方向与步 | 请求数 A→B | Workspace API 次数 A→B | ACL launch 数 A→B | "
           "producer p95（µs） | 队列等待 p95（µs） | consumer p95（µs） | 单次 ACL launch p95（µs） |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, direction in (("forward", "前向"), ("backward", "反向")):
        for step in (4, 5):
            a, b = (result["operators"][name][step][k] for k in ("A", "B"))
            cells = [direction + " #" + str(step), *(numbers(a[k], b[k], 0) for k in ("requests", "workspace_calls", "kernel_launches")),
                     *(numbers(a["latency_us"][k]["p95"], b["latency_us"][k]["p95"]) for k in LATENCY)]
            out.append("| " + " | ".join(cells) + " |")
    out.extend(["", "producer 从 ATen 入口至对应 enqueue 开始。等待从 enqueue 结束至 dequeue 开始，保留有符号的跨线程时间差。"
                "consumer 为 dequeue 服务时长。ACL 分位数按单次实际 kernel launch 统计。[逐笔及长尾](operator_latency.csv)", "",
                "| 方向与步 | 后继设备空档 ms | 相交 workspace ms | 相交等待 ms | 相交 consumer 至 ACL 开始 ms | 相交 ACL ms | ACL 返回后 ms |",
                "|---|---:|---:|---:|---:|---:|---:|"])
    for name, direction in (("forward", "前向"), ("backward", "反向")):
        for step in (4, 5):
            a, b = (result["operators"][name][step][k] for k in ("A", "B"))
            out.append("| " + " | ".join([direction + " #" + str(step)] +
                       [numbers(a["gap_ms"][k], b["gap_ms"][k], 3) for k in GAPS]) + " |")
    out.extend(["", "设备空档按所有计算与搬运流的工作区间取并集，按后继设备工作归入请求。表内给出的 Host 交集"
                "只使用该后继请求的时间段。单调性或重叠不足时不作因果断言。[空档分解](gap_attribution.csv)", "",
                "| 方向与步 | 后继空档数 A→B | 起点落于 workspace | 起点落于可见等待 | 起点已 dequeue、未到 ACL | 起点在 ACL 返回后 |",
                "|---|---:|---:|---:|---:|---:|"])
    for name, direction in (("forward", "前向"), ("backward", "反向")):
        for step in (4, 5):
            a, b = (result["operators"][name][step][k] for k in ("A", "B"))
            out.append("| " + " | ".join([direction + " #" + str(step), numbers(a["gaps_n"], b["gaps_n"], 0)] +
                       [numbers(a["onset_of_successor_gap"].get(k, 0), b["onset_of_successor_gap"].get(k, 0), 0)
                        for k in ("workspace_ns", "visible_queue_wait_ns", "consumer_before_api_ns", "after_api_ns")]) + " |")
    out.extend(["", "起点统计表示后继请求在空档开始时最早可见的阶段。[A 全部阶段](A/first_visible.csv) · [B 全部阶段](B/first_visible.csv)", "",
                "| 步 | 物理流 | 工作项数 A→B | 同名同槽 | 不同名称的替换及数量 |",
                "|---:|---:|---:|---:|---|"])
    for step, details in result["full_step"].items():
        for stream in details["physical_streams"]:
            changed = "，".join(f"{s['a'][1]} → {s['b'][1]} × {s['n']}" for s in stream["substitutions"])
            out.append(f"| {step} | {stream['stream']} | {stream['a_count']} → {stream['b_count']} | "
                       f"{stream['identical_slots']} | {changed or '—'} |")
    out.extend(["", "| 整步 | ACL kernel | 异步 memcpy | 同步 memcpy | Level1 Free ms | Level1 Stage ms | Level1 Computing ms |",
                "|---:|---:|---:|---:|---:|---:|---:|"])
    for step, details in result["full_step"].items():
        a, b = details["A"], details["B"]
        fields = ("aclrtLaunchKernelWithHostArgs", "aclrtMemcpyAsync", "aclrtMemcpy")
        out.append("| " + " | ".join([str(step)] +
                   [numbers(a["acl_submission_apis"].get("AscendCL@" + k, 0),
                            b["acl_submission_apis"].get("AscendCL@" + k, 0), 0) for k in fields] +
                   [numbers(a["level1_ms"][k], b["level1_ms"][k], 3) for k in ("Free", "Stage", "Computing")]) + " |")
    out.extend(["", "这里的 Free 仅描述本次 Level1 归因窗口。正式 Free 占比取独立 Level0 正反序的第 4–6 步。"
                "[全步 ATen 顺序、各类 ACL 提交及物理流核查](submissions_compare.json) · "
                "[各方向匹配、workspace 总时长和 p50/p95/p99/max](embedding_metrics.json)"])
    (data / "RESULTS.md").write_text("\n".join(out) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("raw", type=Path)
    p.add_argument("data", type=Path)
    p.add_argument("--analyzers", required=True, type=Path)
    p.add_argument("--target", required=True, choices=("emb-back", "emb-fwd"))
    args = p.parse_args()
    compare(args.raw, args.data, args.analyzers)
    note(collect(args.raw, args.data), args.data, args.target)


if __name__ == "__main__":
    main()
