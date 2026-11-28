"""Render both paired orders and whole-stream accounting from the verified receipts."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("operator",choices=("aten::view","empty_tensor"))
    args=parser.parse_args()
    data=json.loads((args.directory/"paired-endpoints.json").read_bytes())
    l1=json.loads((args.directory/"level1.json").read_bytes())
    rows=data["rounds"]
    lines=["# 动态 eager 实机配对", "", "下表第一行为正序 A/B，第二行为反序 B/A。时间单位均为 ms/步。",
           "","|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|",
           "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    keys=("baseline_ms","candidate_ms","step_gain_pct","baseline_free_ms","baseline_free_pct","candidate_free_ms","candidate_free_pct","free_ratio_gain_pct")
    for r in rows:
        vals=r["eight_columns"]
        cells=[f"{vals[k]:.3f}"+("%" if k.endswith("_pct") else "") for k in keys]
        lines.append("|"+"|".join(cells)+"|")
    lines+=["","|顺序|腿|样本/s|token/s|Stage ms/步|Computing ms/步|","|---|---|---:|---:|---:|---:|"]
    for r in rows:
        for leg in r["order"].upper():
            t,f=r["steady"][leg],r["level0"][leg]
            lines.append(f"|{r['order'].upper()}|{leg}|{t['samples_per_s']:.4f}|{t['tokens_per_s']:.3f}|"
                         f"{f['Stage_ms_per_step']:.3f}|{f['Computing_ms_per_step']:.3f}|")
    sd=data["paired_delta_ms_per_step"]["sample_sd"]
    lines+=["",f"配对差 A−B 为 {rows[0]['delta_ms_per_step']:+.3f}、{rows[1]['delta_ms_per_step']:+.3f} ms/步，"
            f"两对差值的样本标准差为 {sd:.3f} ms/步。10 步预热后的 50 步首尾同步，"
            "每腿实际处理 136 个样本、2,994 个 token。Level0 第 4–6 步依次为 [1,19]、[2,14]、[2,22]，占比取 ΣFree/ΣStage。"
            "[未舍入计时与分母](paired-endpoints.json)","",
            "## 第 4 和第 5 步的 Level1","","下表每格依次为 A→B。分位数单位为 µs，交集单位为 ms。"
            "“入队前空档”按下一笔真实设备工作关联到唯一的 queue ID 和 producer 线程，再与该线程最内层标记求交。"
            "两步均为对应 [batch, length] 的首次出现。","",
            "|指标|#4 [1,19]|#5 [2,14]|","|---|---:|---:|"]
    def pair(fn,step):
        return "→".join(str(fn(l1[v]["steps"][step])) for v in ("A","B"))
    metrics=[
        ("ATen 到入队 p95",lambda x:f"{x['all_submissions']['latency']['aten_to_enqueue_ns']['p95_us']:.2f}"),
        ("入队 p95",lambda x:f"{x['all_submissions']['latency']['enqueue_ns']['p95_us']:.2f}"),
        ("可见队列等待 p95/p99/最大",lambda x:"/".join(f"{x['all_submissions']['latency']['signed_queue_wait_ns'][v]:.2f}" for v in ("p95_us","p99_us","max_us"))),
        ("consumer 服务 p95",lambda x:f"{x['all_submissions']['latency']['dequeue_ns']['p95_us']:.2f}"),
        ("单次 ACL kernel launch p95",lambda x:f"{x['launch_us']['p95']:.2f}"),
        ("全步 ACL kernel launch 次数",lambda x:x["acl"]["AscendCL@aclrtLaunchKernelWithHostArgs"]),
        ("同步/异步 ACL memcpy 次数",lambda x:f"{x['acl'].get('AscendCL@aclrtMemcpy',0)}/{x['acl'].get('AscendCL@aclrtMemcpyAsync',0)}"),
        ("逐流物理工作数 44/45/47",lambda x:"/".join(str(x["physical"].get(i,{}).get("work_n",0)) for i in ("44","45","47"))),
        (args.operator+" 次数",lambda x:x["ops"][args.operator]["calls"]),
        (args.operator+" p50/p95/p99/最大",lambda x:"/".join(f"{x['ops'][args.operator]['latency_us'][v]:.2f}" for v in ("p50","p95","p99","max"))),
        (args.operator+" 独占 CPU ms",lambda x:f"{x['ops'][args.operator]['exclusive_cpu_ms']:.3f}"),
        (args.operator+" 入队前空档交集 ms",lambda x:f"{x['ops'][args.operator]['before_enqueue_gap_ms']:.3f}"),
    ]
    for name,fn in metrics:
        lines.append("|"+name+"|"+"|".join(pair(fn,s) for s in ("4","5"))+"|")
    lines+=["","|step|ATen 计数与逐线程顺序均相同|queue 计数相同|逐流设备工作顺序相同|同流同位置工作类型与名称相同|",
            "|---|---|---|---|---:|"]
    yes=lambda b:"是" if b else "否"
    for step,checks in l1["compare"].items():
        slots=checks["physical_slots"]
        lines.append(f"|{step}|{yes(checks['same_aten_counts'] and checks['same_aten_per_thread_order'])}|"
                     f"{yes(checks['same_enqueue_counts'])}|{yes(checks['same_physical_streams_and_work_order'])}|"
                     f"{slots['identical']:,}/{slots['paired']:,}|")
    lines+=["","原始 L0、L1、核详情、运行命令、调度形状及逐卡采样位于证据包。"
            "未舍入 JSON 保留各腿物理流哈希、各阶段长尾和工作位置差异。"
            "[完整配对分位数及流校验](level1.json) · [原始证据](evidence.tar.gz) · [设备独占校验](slot.json)",""]
    (args.directory/"实机结果.md").write_text("\n".join(lines),encoding="utf-8")


if __name__=="__main__": main()
