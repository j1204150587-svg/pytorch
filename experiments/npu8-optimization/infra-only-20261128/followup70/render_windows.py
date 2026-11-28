"""Render the verified direct A/B and B/A training windows for follow-up candidates."""
import json
from pathlib import Path
import sys

def render(root):
    measurements=json.loads((root/"metrics.json").read_bytes())
    rows=measurements["rounds"]
    assert [r["order"] for r in rows]==["ab","ba"]
    lines=["# 原动态训练双序结果","","两腿工作量、训练配置和运行顺序见原始命令。前 10 步预热，第 10–59 步的同步窗口处理 136 个样本、2,994 个 token。Level0 在 [1,19]、[2,14]、[2,22] 三步复算 ΣFree/ΣStage。以下时间均为 ms/步，首行为正序，次行为反序，两行均按 A→B 展示。","",
        "|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|",
        "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        x=r["eight_columns"]
        lines.append("|{baseline_ms:.3f}|{candidate_ms:.3f}|{step_gain_pct:+.3f}%|{baseline_free_ms:.3f}|{baseline_free_pct:.3f}%|{candidate_free_ms:.3f}|{candidate_free_pct:.3f}%|{free_ratio_gain_pct:+.3f}%|".format(**x))
    lines+=["","|顺序|腿|样本/s|token/s|Stage ms/步|Computing ms/步|","|---|---|---:|---:|---:|---:|"]
    for r in rows:
        for arm in r["order"].upper():
            s,lev=r["steady"][arm],r["level0"][arm]
            lines.append(f"|{r['order'].upper()}|{arm}|{s['samples_per_s']:.3f}|{s['tokens_per_s']:.3f}|{lev['Stage_ms_per_step']:.3f}|{lev['Computing_ms_per_step']:.3f}|")
    deltas=[r["steady"]["A"]["ms_per_step"]-r["steady"]["B"]["ms_per_step"] for r in rows]
    d=measurements["paired_delta_ms_per_step"]
    lines+=["",f"两序步时差（A 减 B）为 {deltas[0]:+.3f}、{deltas[1]:+.3f} ms/步，均值 {d['mean']:+.3f}，样本标准差 {d['sample_sd']:.3f}。每腿卡况、加载库与原始 CSV 已随实机数据核验。[逐腿复算](metrics.json) · [原始窗口与命令](windows-and-commands.tgz) · [完整原始归档](raw.tgz) · [逐请求与物理流](summary.tgz) · [原始宿主目录及指纹](evidence-index.json)",""]
    (root/"双序结果.md").write_text("\n".join(lines),encoding="utf-8")

if __name__=="__main__":
    for path in sys.argv[1:]:
        render(Path(path))
