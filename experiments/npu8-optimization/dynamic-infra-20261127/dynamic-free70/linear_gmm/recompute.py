"""Summarize shape eligibility, BF16 equality and timing from completed case receipts."""
import argparse
import json
from pathlib import Path
from statistics import median


def physical(prof):
    return [(e["stream"],e["name"],e["n"]) for e in prof.get("physical",[])
            if e["kind"].startswith(("AI_","MIX_","SDMA","HDMA"))]


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("directory",type=Path)
    cli.add_argument("output",type=Path)
    args=cli.parse_args()
    rows=[]
    for fn in sorted(args.directory.glob("b*-l*-i*-o*.json")):
        j=json.loads(fn.read_text(encoding="utf8"))
        timings=j.get("window_us_per_pair",[])
        row=dict(shape=j["shape"],k=j["M"],projection=[j["cin"],j["cout"]],status=j["status"],
                 bytes_for_transpose=j.get("transpose_copy_bytes"),error=j.get("error"))
        for k in ("stock_vs_manual","stock_vs_grouped"):
            if k in j:
                row[k]={p:v.get("different","unmatched shapes") for p,v in j[k].items()}
        if timings:
            row.update(median_a_us=median(x["A"] for x in timings),median_b_us=median(x["B"] for x in timings),
                       net_saving_per_pair_us=[round(x["A"]-x["B"],3) for x in timings])
        if j.get("profiles"):
            row["physical_tasks_by_stream"]={k:physical(v) for k,v in j["profiles"].items()}
        rows.append(row)
    result=dict(cases=rows,completed=sum(r["status"]=="completed" for r in rows),
                exact=sum(bool(r.get("stock_vs_grouped")) and all(v==0 for v in r["stock_vs_grouped"].values()) for r in rows))
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
