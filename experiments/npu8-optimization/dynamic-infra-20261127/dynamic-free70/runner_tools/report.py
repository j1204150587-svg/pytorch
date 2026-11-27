"""Recompute the dynamic-eager Free target and its paired throughput from original per-leg receipts."""
import argparse
import json
from pathlib import Path
import sys

from run import HERE, STUDY, FREE70, check, read, recipe_info
from verify import verify_activation

sys.path.insert(0,str(STUDY/"reporting"))
from recompute_pairs import parse_pair, summarize, markdown, write_tables, measurement_tables


def validate_controls(leg,recipe):
    path = Path(leg["directory"])
    rank = path/"rank000"
    env = leg["environment"]
    expected = {**recipe["common"],**recipe[leg["variant"]]}
    check(env is not None and all(env.get(k)==v for k,v in expected.items()),f"{path}: recipe mismatch")
    steps = 60 if "ms_per_step" in leg else 8
    for name,filename in (("PARTITION","fast_partition.json"),("NORM","norm_stats.json")):
        enabled = any(env.get(key)=="1" for key in ("RECSYS_DYNAMIC_"+name,"RECSYS_DYNAMIC_BOTH"))
        p = rank/filename
        check(p.is_file()==enabled,f"{path}: missing/unexpected {filename}")
        if enabled:
            obj=read(p)
            if name=="NORM":
                check(obj["valid"] and obj["calls"]==obj["batched_calls"]==steps and not obj["fallback_reasons"],
                      f"{path}: batched norm counters")
            else:
                check(obj["partition"]==steps,f"{path}: partition count")
    if recipe.get("comparison") != "stock01-to-combination" or leg["variant"] == "B":
        verify_activation(recipe,leg["variant"],rank,steps)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("recipe",type=Path)
    cli.add_argument("--receipts",type=Path,help="read extracted <recipe>-<mode>-<order> directories instead of host registry")
    cli.add_argument("--schedule",type=Path,help="explicit 60-shape JSON, e.g. from an extracted workspace")
    cli.add_argument("--output",type=Path)
    args = cli.parse_args()
    recipe,switch=recipe_info(args.recipe)
    name=args.recipe.stem
    work=FREE70/"runs"/name
    schedule_path=args.schedule or work/"runner/dynamic_shapes_npu8.json"
    shapes=read(schedule_path)
    check(len(shapes)==len(set(map(tuple,shapes)))==60 and sum(b for b,n in shapes[10:])==136 and
          sum(b*n for b,n in shapes[10:])==2994, "unexpected 60-shape input schedule")
    if args.receipts:
        entries = [{"mode":m,"order":o,"directory":str(args.receipts/f"{name}-{m}-{o}")}
                   for m in ("l0","l1","no-profiler") for o in ("baseline","aa","ab","ba")
                   if (args.receipts/f"{name}-{m}-{o}"/"status.json").is_file()]
    else:
        entries=[e for e in read(work/"runs.json") if e["exit"]==0]
    check(entries,"no completed runs")
    pairs=[]
    for e in entries:
        order=e["order"] if e["order"]!="baseline" else "a"
        endpoint = recipe.get("comparison") == "stock01-to-combination"
        pair = parse_pair(f'r1/{order}/{e["mode"]}={e["directory"]}',shapes,None if endpoint else switch)
        if endpoint and len(pair["legs"]) > 1:
            before, after = recipe["A"], recipe["B"] if order != "aa" else recipe["A"]
            expected_changes = {key: [before.get(key), after.get(key)] for key in before.keys() | after.keys()
                                if before.get(key) != after.get(key)}
            check(pair["changed_environment"] == expected_changes,
                  f'{e["directory"]}: environment differences must match the declared endpoint recipe')
        for leg in pair["legs"]:
            validate_controls(leg,recipe)
        pairs.append(pair)
    rows,variation=summarize(pairs)
    orders={row["order"]:row["cells"] for row in rows}
    completed={"AB","BA"}==orders.keys()
    target={"weighted_B_Free_share_below_0.70_both_orders": None,
            "unprofiled_e2e_faster_both_orders": None,
            "absolute_Free_time_lower_both_orders": None,
            "paired_measurements_meet_free_and_e2e_target": None}
    if completed:
        target["weighted_B_Free_share_below_0.70_both_orders"]=all(v[6]<0.70 for v in orders.values())
        target["unprofiled_e2e_faster_both_orders"]=all(v[1]<v[0] for v in orders.values())
        target["absolute_Free_time_lower_both_orders"]=all(v[5]<v[3] for v in orders.values())
        target["paired_measurements_meet_free_and_e2e_target"]=(target["weighted_B_Free_share_below_0.70_both_orders"]
                                                               and target["unprofiled_e2e_faster_both_orders"]
                                                               and target["absolute_Free_time_lower_both_orders"])
    output=args.output or work/"report"
    output.mkdir(parents=True,exist_ok=True)
    write_tables(output,pairs)
    (output/"paired_results.json").write_text(json.dumps({"recipe":recipe,"schedule":str(schedule_path),"pairs":pairs,
           "eight_column_rows":rows,"variation":variation,"target":target},indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    (output/"measurement.md").write_text(measurement_tables(pairs,rows,{"r1":variation}),encoding="utf-8")
    (output/"eight_columns.md").write_text(markdown(rows),encoding="utf-8")
    baseline={}
    for pair in pairs:
        if pair["order"]=="A":
            leg=pair["legs"][0]
            baseline[pair["mode"]]={k:leg[k] for k in ("ms_per_step","samples_per_s","tokens_per_s",
                       "free_ms","stage_ms","computing_ms","free_fraction") if k in leg}
    print(json.dumps({"baseline":baseline,"eight_column_rows":rows,"target":target},indent=2))


if __name__=="__main__":
    main()
