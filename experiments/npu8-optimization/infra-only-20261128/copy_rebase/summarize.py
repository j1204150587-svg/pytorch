"""Recheck paired metrics and generate compact markdown tables directly from the native receipts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    p.add_argument("prefix")
    p.add_argument("out",type=Path)
    args=p.parse_args()
    root=Path(__file__).resolve().parent.parent
    schedule=root/"evidence/archived-runner/dynamic_shapes_npu8.json"
    command=[sys.executable,"-B",str(root/"runner/paired_metrics.py"),"--schedule",str(schedule),
             "--raw-root",str(args.folder/"raw")]
    for order in ("ab","ba"):
        command.extend(["--round",str(args.folder/"pairs"/f"{args.prefix}-no-profiler-{order}"),
                        str(args.folder/"pairs"/f"{args.prefix}-l0-{order}")])
    metrics=json.loads(subprocess.check_output(command))
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/"paired.json").write_text(json.dumps(metrics,indent=2)+"\n")
    text=["# "+args.prefix+" 原始训练交叉序结果","",
        "每腿稳定窗口为 10 步预热后的 50 步，共 136 个样本和 2,994 个 token。每个表单独列出 A/B、B/A 排序。端到端提升按 (A−B)/A，Free 占比按三步 ΣFree/ΣStage，Free 占比提升按 (占比A−占比B)/占比A。",""]
    for r in metrics["rounds"]:
        title=r["order"].upper()
        c=r["eight_columns"]
        text.extend(["## "+title,"","| 优化前端到端 ms/步 | 优化后端到端 ms/步 | 端到端提升 | 优化前 Free ms/步 | 优化前 Free 占比 | 优化后 Free ms/步 | 优化后 Free 占比 | Free 占比提升 |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| "+" | ".join(f"{c[n]:.3f}"+("%" if n.endswith("pct") else "") for n in
                ("baseline_ms","candidate_ms","step_gain_pct","baseline_free_ms","baseline_free_pct",
                 "candidate_free_ms","candidate_free_pct","free_ratio_gain_pct"))+" |","",
            "| 训练腿 | 样本/s | token/s | Stage ms/步 | Computing ms/步 |","|---|---:|---:|---:|---:|"])
        for v in "AB":
            st,prof=r["steady"][v],r["level0"][v]
            text.append(f"| {v} | {st['samples_per_s']:.3f} | {st['tokens_per_s']:.3f} | {prof['Stage_ms_per_step']:.3f} | {prof['Computing_ms_per_step']:.3f} |")
        text.append("")
    fluct=metrics["paired_delta_ms_per_step"]
    text.append("A−B 的两组绝对差为 "+", ".join(f"{r['delta_ms_per_step']:+.3f}" for r in metrics["rounds"])+
        f" ms/步，均值 {fluct['mean']:+.3f} ms/步，样本标准差 {fluct['sample_sd']:.3f} ms/步。")
    (args.out/"PAIRS.md").write_text("\n".join(text)+"\n",encoding="utf-8")


if __name__=="__main__":
    main()
