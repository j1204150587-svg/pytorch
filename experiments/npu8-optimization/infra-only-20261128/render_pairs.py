"""Render a standalone paired report from the unrounded window and profiler receipts."""
import argparse
import json
from pathlib import Path

NAMES={"addmm":"前向 Addmm 描述符","mm":"反向 Mm 描述符","divs":"侧表反向 Div 广播",
       "queue":"Task Queue 满环等待者唤醒","emb-back":"大缓存下增加 embedding 反向描述符规范化",
       "emb-fwd":"大缓存和反向规范化下增加 embedding 前向规范化"}

def render(path):
    data=json.loads(path.read_bytes())
    name=path.stem
    rows=data["rounds"]
    out=[f"# {NAMES[name]}", "", "第一行为正序 A/B，第二行为反序 B/A，表内均按 A→B 展示。时间单位为 ms/步。","",
         "|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|",
         "|---:|---:|---:|---:|---:|---:|---:|---:|"]
    keys=("baseline_ms","candidate_ms","step_gain_pct","baseline_free_ms","baseline_free_pct","candidate_free_ms","candidate_free_pct","free_ratio_gain_pct")
    for r in rows:
        cols=r["eight_columns"]
        out.append("|"+"|".join(f"{cols[k]:.3f}" if k.endswith("_ms") else f"{cols[k]:+.2f}%" if "gain" in k else f"{cols[k]:.2f}%" for k in keys)+"|")
    out.extend(["","|顺序|腿|样本/s|token/s|Level0 Free ms/步|Stage ms/步|Computing ms/步|",
                "|---|---|---:|---:|---:|---:|---:|"])
    links=[]
    for r in rows:
        for which in r["order"].upper():
            s,lv=r["steady"][which],r["level0"][which]
            out.append(f"|{r['order'].upper()}|{which}|{s['samples_per_s']:.3f}|{s['tokens_per_s']:.3f}|{lv['Free_ms_per_step']:.3f}|{lv['Stage_ms_per_step']:.3f}|{lv['Computing_ms_per_step']:.3f}|")
        for mode, k in (("无 profiler","steady"),("Level0","level0")):
            raw=r[k]["directory"].replace("\\","/").rstrip("/").split("/")[-1]
            links.append(f"[{r['order'].upper()} {mode}](../evidence/raw/{raw}/status.json)")
    sd=data["paired_delta_ms_per_step"]["sample_sd"]
    a,b=rows
    out.extend(["",f"两序 A−B 分别为 {a['delta_ms_per_step']:+.3f}、{b['delta_ms_per_step']:+.3f} ms/步，两个配对差的样本标准差为 {sd:.3f} ms/步。每腿无 profiler 窗口共 136 个样本、2,994 个 token。Level0 三步依次为 `[1,19]`、`[2,14]`、`[2,22]`。Free 占比用三步的 ΣFree/ΣStage。","",
                f"该项开关为 `{a['steady']['recipe']['switch']}`，原生库、额外前置及分母保存在复算数据中。[未舍入数据]({name}.json) · "+" · ".join(links),""])
    path.with_suffix(".md").write_text("\n".join(out),encoding="utf-8")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("reports",nargs="+",type=Path)
    for path in ap.parse_args().reports: render(path)

