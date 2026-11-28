"""Recompute synchronized training windows, Level0 device time, and Level1 submission chains."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import os
import re
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path


MODES = ("no-profiler", "l0", "l1")
ORDERS = ("ab", "ba")
COMPARISONS = ("fixed-cache", "stock-to-full", "default-cache")
EIGHT = ("baseline_ms", "candidate_ms", "step_gain_pct", "baseline_free_ms", "baseline_free_pct",
         "candidate_free_ms", "candidate_free_pct", "free_ratio_gain_pct")
TRAIN_FLAGS = ("--seed", "--model-config", "--deepspeed-config", "--shape-schedule", "--micro-batch-size",
               "--learning-rate", "--weight-decay")


def check(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_bytes().decode("utf-8-sig"))


def digest(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def local_path(base, value):
    path = (base / value).resolve()
    if os.name == "nt" and not str(path).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(path))
    return path


def write(path, data):
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=path.parent,
                                     prefix=path.name+".", delete=False) as f:
        tmp = Path(f.name)
        f.write(data)
    tmp.replace(path)


def json_text(data):
    return json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n"


def spread(values):
    if not values:
        return {"n": 0}
    center = statistics.median(values)
    return dict(n=len(values), median=center, mean=statistics.mean(values), min=min(values), max=max(values),
                sample_sd=statistics.stdev(values) if len(values)>1 else None,
                mad=statistics.median(abs(v-center) for v in values))


def gain(a,b):
    return (1-b/a)*100 if a else None


def flag(argv, name):
    check(argv.count(name)==1 and argv.index(name)+1<len(argv), f"missing/duplicate training argument {name}")
    return argv[argv.index(name)+1]


def validate_leg(leg, mode, schedule):
    command=read(leg/"command.json")
    check(command["variant"]==leg.name[-1] and command["mode"]==mode, f"leg/mode mismatch: {leg}")
    argv = command["argv"]
    settings={key:flag(argv,key) for key in TRAIN_FLAGS}
    raw=read(leg/"rank000/metrics.json")
    steps=raw["steps"]
    shapes=[[r["batch_size"],r["seq_len"]] for r in steps]
    check([r["step"] for r in steps]==list(range(len(steps))) and shapes==schedule[:len(steps)],
          f"actual shape sequence mismatch: {leg}")
    check(len(steps)==int(flag(argv,"--steps")), f"incomplete steps: {leg}")
    check(("--profile" in argv)==(mode!="no-profiler"), f"wrong profiler setting: {leg}")
    check(("--profile-level1" in argv)==(mode=="l1"),f"wrong profiler level: {leg}")
    if mode=="no-profiler":
        check((flag(argv,"--steady-warmup-steps"),flag(argv,"--steady-measure-steps"))==("10","50"),
              f"expected 10+50: {leg}")
    return raw,settings


def slot_proof(receipt, raw, windows, order):
    truth=read(receipt/"slot-verified.json")
    records=[json.loads(v) for v in windows.read_bytes().splitlines() if v]
    check(len({r["utc"] for r in records})==len(records),"duplicate chip timestamp")
    result={}
    for i,arm in enumerate(order.upper(),1):
        leg=f"{i:02}-{arm}"
        expected=read(receipt/leg/"worker.json")["pid"]
        before,after=[read(raw/leg/(pos+".card.json")) for pos in ("before","after")]
        for card in (before,after):
            check([card[k] for k in ("logical_device","board","chip","physical_id","health","host_pids")]==[8,4,0,8,"OK",[]],
                  f"invalid NPU 8 device mapping or existing context: {raw/leg}")
        start=datetime.fromisoformat(before["utc"])
        end=datetime.fromisoformat(after["utc"])+timedelta(seconds=1)
        selected=[r for r in records if start<=datetime.fromisoformat(r["utc"])<end]
        check(selected and all(r["valid"] for r in selected),"invalid or missing chip samples")
        check(all(p["pid"]==expected and p["owner"]=="training" for r in selected for p in r["processes"]),
              f"unexpected device process: {raw/leg}")
        hit=sum(bool(r["processes"]) for r in selected)
        target=truth["legs"][leg]
        check(hit and (target["worker"],target["samples"],target["with_worker"])==(expected,len(selected),hit),
              f"host slot verdict differs: {raw/leg}")
        result[arm]=dict(worker=expected,samples=len(selected),with_worker=hit,begin=before["utc"],end=after["utc"])
    return result


def proof(row, root, raw, order, mode, comparison):
    if "receipt" not in row:
        return {"status":"raw_only"}
    receipt = local_path(root,row["receipt"])
    entry=read(receipt/"result.json")
    audit=mode=="audit"
    check(entry["exit"]==0 and entry["order"]==order and entry["mode"]==("no-profiler" if audit else mode)
          and bool(entry["audit"])==audit,
          f"invalid host pair: {receipt}")
    check(Path(entry["directory"]).name==raw.name, f"receipt points to different raw pair: {receipt}")
    recipe=read(receipt/"recipe.json") if (receipt/"recipe.json").exists() else None
    result=dict(status="host_receipt",path=row["receipt"],recipe=recipe,legs={})
    if comparison in COMPARISONS:
        check("chip_window" in row and recipe and set(recipe)=={"endpoints"},f"missing chip samples/endpoints: {receipt}")
        result["chip"]=slot_proof(receipt,raw,local_path(root,row["chip_window"]),order)
        result["chip_sha256"]=digest(local_path(root,row["chip_window"]))
        check(len(recipe["endpoints"])==2,f"expected two endpoint recipes: {receipt}")
    for i,arm in enumerate(order.upper(),1):
        folder=receipt/f"{i:02}-{arm}"
        worker=read(folder/"worker.json")
        dispatch=read(folder/"dispatch.json")
        check(worker["pid"]>0 and worker["libraries"],f"missing native mapping: {folder}")
        spec=recipe["endpoints"][arm] if recipe and "endpoints" in recipe else recipe
        if spec and {"common","switch","A","B","libraries"} <= spec.keys():
            expected={**spec["common"],spec["switch"]:spec[arm]}
            for key,value in expected.items():
                actual=worker["env"].get(key)
                value=value.replace("{output}",dispatch["output"])
                if key=="PYTHONPATH":
                    check(value in (actual or "").split(":"),f"native PYTHONPATH: {folder}")
                else:
                    check(actual==value,f"{folder}: {key} differs")
            check(set(worker["libraries"])==set(spec["libraries"]),f"library set differs: {folder}")
            check(dispatch["libraries"]==spec["libraries"],f"library SHA receipt differs: {folder}")
        if comparison in COMPARISONS:
            flags=worker["actual_flags"]
            switches=("NPU_DYNAMIC_REBASE_INDICES","NPU_DYNAMIC_REBASE_FORWARD","TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY")
            stock=(comparison=="stock-to-full" and arm=="A")
            active=None if stock else str(int(arm=="B"))
            check(all(flags[k]==active for k in switches),f"unexpected native switch vector: {folder}")
            cache="65536" if comparison!="default-cache" and not stock else None
            counted=audit and comparison=="fixed-cache"
            counter=None if stock else str(int(counted))
            check(flags["ACLNN_CACHE_LIMIT"]==cache and all(flags[k]==counter for k in
                  ("NPU_DYNAMIC_REBASE_COUNT","TORCH_NPU_QUEUE_WAKE_AUDIT")),f"cache/counter drift: {folder}")
            check(("INFRA_COUNTER_DIR" in expected)==counted and ("LD_PRELOAD" in expected)==counted,
                  f"unexpected profiler counter/preload setting: {folder}")
        result["legs"][arm]=dict(worker_pid=worker["pid"],mapped_libraries=sorted(worker["libraries"]),
                                 dispatch_sha256=digest(folder/"dispatch.json"),worker_sha256=digest(folder/"worker.json"),
                                 worker_env=worker.get("env",{}),actual_flags=worker.get("actual_flags"),
                                 library_digests=dispatch.get("libraries",{}))
    return result


def steady(raw, schedule, folder):
    w=raw["summary"]["steady_state"]
    check((w["start_step_inclusive"],w["end_step_exclusive"],w["steps"],len(raw["steps"]))==(10,60,50,60),
          f"bad synchronized window: {folder}")
    samples=sum(b for b,l in schedule[10:60])
    tokens=sum(b*l for b,l in schedule[10:60])
    check((w["samples"],w["tokens"],w["loss_log_policy"])==(samples,tokens,"deferred_until_after_end_sync"),
          f"workload/endpoint mismatch: {folder}")
    sec=w["elapsed_s_synchronized"]
    check(math.isfinite(sec) and sec>0,f"invalid synchronized elapsed time: {folder}")
    return dict(ms=sec*1000/50,seconds=sec,samples=samples,tokens=tokens,
                samples_per_s=samples/sec,tokens_per_s=tokens/sec)


def level0(folder, schedule, old):
    # This existing validator checks the raw step order, device, shape and units.
    data=old.leg_result(folder,schedule,"l0")
    with (folder/"step_trace_time.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    steps=[]
    for src in rows:
        step=int(src["Step"])
        values={k:float(src[k])/1000 for k in ("Free","Stage","Computing")}
        check(values["Stage"]>0 and values["Free"]<=values["Stage"]+1e-6, f"bad Free/Stage: {folder}")
        steps.append(dict(step=step,shape=schedule[step],**values))
    return dict(steps=steps,free_ms=data["Free_ms_per_step"],stage_ms=data["Stage_ms_per_step"],
                computing_ms=data["Computing_ms_per_step"],free_pct=data["free_ratio"]*100)


def linked_csv(path):
    with gzip.open(path,"rt",newline="",encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def by_op_and_stream(linked):
    calls=linked_csv(linked/"queue_calls.csv.gz")
    tasks=linked_csv(linked/"work_tasks.csv.gz")
    info={(r["step"],r["queue_id"]):r["op"] for r in calls}
    groups=defaultdict(lambda:dict(requests=0,acl_submissions=0,by_stream=defaultdict(Counter)))
    for r in calls:
        g=groups[(r["step"],r["op"])]
        g["requests"]+=1
        g["acl_submissions"]+=int(r["launch_api_count"])
    for r in tasks:
        op=info.get((r["step"],r["queue_id"]),"<unlinked>")
        groups[(r["step"],op)]["by_stream"][r["stream"]][r["task"]]+=1
    selected=("aclnnEmbedding","aclnnEmbeddingDenseBackward")
    per_request=defaultdict(list)
    for r in tasks:
        per_request[r["step"],r["queue_id"]].append((r["stream"],r["task_type"],r["task"]))
    anchors=defaultdict(dict)
    for r in calls:
        if r["op"] not in selected or not r["aten"] or not r["aten_ordinal_in_thread"]:
            continue
        key=(r["role"],r["aten"],int(r["aten_ordinal_in_thread"]))
        index=anchors[(r["step"],r["op"])]
        check(key not in index,f"nonunique target ATen anchor: {key}")
        seq=tuple(per_request[r["step"],r["queue_id"]])
        index[key]=(r["acl_api_sequence"],seq)
    return {step:{op:v for (s,op),v in groups.items() if s==step} for step in ("4","5")},anchors


def compare_embedding(a,b,step):
    matches={}
    for op in ("aclnnEmbedding","aclnnEmbeddingDenseBackward"):
        left,right=a[(str(step),op)],b[(str(step),op)]
        shared=left.keys() & right.keys()
        result=Counter()
        for k in shared:
            acl_a,work_a=left[k]
            acl_b,work_b=right[k]
            if not work_a or not work_b:
                result["missing_device_link"]+=1
                continue
            result["both_have_device_links"]+=1
            result["same_acl_sequence"]+=acl_a==acl_b
            result["same_streams_and_task_counts"]+=Counter(x[0] for x in work_a)==Counter(x[0] for x in work_b)
            result["same_stream_task_types_and_names"]+=work_a==work_b
        matches[op]=dict(A_anchors=len(left),B_anchors=len(right),shared=len(shared),**result)
    return matches


def level1(raw, dest, study, schedule):
    import analyze_pair
    analyzer=study/"followup70/producer-sched/analyze_dynamic_trace.py"
    checks=[digest(analyzer),digest(study/"queue/analyze_pair.py"),digest(study/"queue/exposure.py"),
            digest(Path(__file__))]
    legs={p.name[-1]:p for p in raw.glob("0?-?") if p.is_dir() and p.name[-1] in "AB"}
    check(set(legs)==set("AB"),f"incomplete Level1: {raw}")
    for arm in "AB":
        p=legs[arm]/"trace_view.json.gz"
        if not p.exists():
            matches=list(legs[arm].rglob("trace_view.json"))
            check(len(matches)==1,f"missing/ambiguous Level1 trace: {legs[arm]}")
            p=matches[0]
        checks += [digest(p),digest(legs[arm]/"rank000/metrics.json")]
    key=hashlib.sha256("|".join(checks).encode()).hexdigest()
    final=dest/key
    cached=final/"pair.json"
    if cached.is_file():
        return read(cached),key
    dest.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest,prefix="working-") as tmp:
        root=Path(tmp)
        pair={}
        anchored={}
        for arm in "AB":
            leg_result=analyze_pair.legdata(legs[arm],root/arm,analyzer)
            check(set(leg_result["steps"])=={4,5},f"expected #4 and #5: {raw}")
            work,anchored[arm]=by_op_and_stream(root/arm/"linked")
            for step,record in leg_result["steps"].items():
                check(record["shape"]==schedule[step] and record["first_seen"]==schedule.index(schedule[step]),
                      f"bad step/first-seen label: {raw}")
                record["request_and_device_work"]=work[str(step)]
                linked_work_count=sum(n for r in work[str(step)].values() for stream in r["by_stream"].values() for n in stream.values())
                check(linked_work_count==sum(r["work_n"] for r in record["physical"].values()),f"unaccounted physical work: {raw}")
            pair[arm]=leg_result
        pair["compare"]={s:analyze_pair.compare(r,pair["B"]["steps"][s]) for s,r in pair["A"]["steps"].items()}
        for step,result in pair["compare"].items():
            result["embedding_same_request"]=compare_embedding(anchored["A"],anchored["B"],step)
        (root/"pair.json").write_text(json_text(pair),encoding="utf-8")
        root.rename(final)
    return read(cached),key


def aggregate(pairs,order):
    subset=[r for r in pairs if order=="all" or r["order"]==order]
    timing=[r for r in subset if r["mode"]=="no-profiler"]
    l0=[r for r in subset if r["mode"]=="l0"]
    sums={}
    for arm in "AB":
        ts=[r[arm] for r in timing]
        rows=[s for r in l0 for s in r[arm]["steps"]]
        sums[arm]=dict(time_ms=spread([v["ms"] for v in ts]),
                       samples_per_s=spread([v["samples_per_s"] for v in ts]),
                       tokens_per_s=spread([v["tokens_per_s"] for v in ts]))
        if rows:
            sums[arm].update(free_ms=statistics.mean(v["Free"] for v in rows),
                             computing_ms=statistics.mean(v["Computing"] for v in rows),
                             stage_ms=statistics.mean(v["Stage"] for v in rows),
                             free_pct=100*sum(v["Free"] for v in rows)/sum(v["Stage"] for v in rows))
    delta=[r["delta_ms"] for r in timing]
    out=dict(timing_pairs=len(timing),level0_pairs=len(l0),legs=sums,
             paired_ms=spread(delta),positive_e2e_pairs=sum(x>0 for x in delta),
             paired_e2e_gain_pct=spread([r["gain_pct"] for r in timing]),
             paired_free_ratio_gain_pct=spread([r["free_ratio_gain_pct"] for r in l0 if r["free_ratio_gain_pct"] is not None]),
             paired_free_ms=spread([r["A"]["free_ms"]-r["B"]["free_ms"] for r in l0]))
    if timing and l0:
        a,b=sums["A"],sums["B"]
        ma,mb=a["time_ms"]["median"],b["time_ms"]["median"]
        out["eight"]=dict(zip(EIGHT,(ma,mb,gain(ma,mb),a["free_ms"],a["free_pct"],b["free_ms"],b["free_pct"],gain(a["free_pct"],b["free_pct"]))))
    return out


def run(manifest,out,study):
    sys.path.insert(0,str(study/"runner"))
    sys.path.insert(0,str(study/"queue"))
    import paired_metrics as old
    cfg=read(manifest)
    base=manifest.parent
    schedule_path=local_path(base,cfg["schedule"])
    schedule=read(schedule_path)
    check(len(schedule)==60 and all(isinstance(s,list) and len(s)==2 and all(type(n) is int and n>0 for n in s) for s in schedule),
          "expected 60 positive [batch,length] input shapes")
    schedule_hash=digest(schedule_path)
    comparison=cfg.get("comparison",cfg["candidate"])
    pairs=[]
    seen_ids=set()
    seen_raw=set()
    signatures=set()
    recipe_digests=set()
    settings=None
    for row in cfg["pairs"]:
        pair_id,order,mode=row["id"],row["order"],row["mode"]
        check(re.fullmatch(r"[A-Za-z0-9_.-]+",pair_id) and pair_id not in (".","..") and pair_id not in seen_ids,
              f"bad or duplicate pair id: {pair_id}")
        seen_ids.add(pair_id)
        check(order in ORDERS and mode in MODES,f"invalid order/mode: {pair_id}")
        raw=local_path(base,row["raw"])
        check(raw not in seen_raw,f"raw run reused: {raw}")
        seen_raw.add(raw)
        plan,status=read(raw/"plan.json"),read(raw/"status.json")
        check(plan["shape_schedule_sha256"]==schedule_hash and plan["stage"]==order and plan["mode"]==mode
              and not plan["audit_state"] and status["exit"]==0,f"raw plan/status failed: {raw}")
        check(plan["order"]==list(order.upper()),f"declared and executed order differ: {raw}")
        evidence=proof(row,base,raw,order,mode,comparison)
        if comparison in COMPARISONS and "branch_library_sha256" in cfg and "legs" in evidence:
            for arm,data in evidence["legs"].items():
                libs=[v for path,v in data["library_digests"].items() if path.endswith("/libtorch_npu.so")]
                stock=comparison=="stock-to-full" and arm=="A"
                check(len(libs)==1 and (libs[0]==cfg["branch_library_sha256"])!=stock,
                      f"branch build/worker library identity differs: {pair_id} {arm}")
        if evidence.get("recipe"):
            recipe_digests.add(json.dumps(evidence["recipe"],sort_keys=True))
        rec=dict(id=pair_id,order=order,mode=mode,raw=row["raw"],proof=evidence,metrics_sha256={},plan_sha256=digest(raw/"plan.json"))
        for i,arm in enumerate(order.upper(),1):
            folder=raw/f"{i:02}-{arm}"
            check((folder/"train.exit").read_text().strip()=="0",f"training failed: {folder}")
            data,leg_settings=validate_leg(folder,mode,schedule)
            if settings is None:
                settings=leg_settings
            check(settings==leg_settings,f"training inputs/settings drifted: {folder}")
            rec["metrics_sha256"][arm]=digest(folder/"rank000/metrics.json")
            if mode=="no-profiler":
                rec[arm]=steady(data,schedule,folder)
            elif mode=="l0":
                rec[arm]=level0(folder,schedule,old)
        signature=(mode,plan.get("utc"),tuple(sorted(rec["metrics_sha256"].items())))
        check(signature not in signatures,f"duplicate receipt content: {pair_id}")
        signatures.add(signature)
        if mode=="no-profiler":
            rec.update(delta_ms=rec["A"]["ms"]-rec["B"]["ms"],gain_pct=gain(rec["A"]["ms"],rec["B"]["ms"]))
        elif mode=="l0":
            rec["free_ratio_gain_pct"]=gain(rec["A"]["free_pct"],rec["B"]["free_pct"])
        else:
            pair,key=level1(raw,out/"level1"/pair_id,study,schedule)
            rec["level1_file"]=f"level1/{pair_id}/{key}/pair.json"
            rec["level1_compare"]=pair["compare"]
        pairs.append(rec)
    check(len(recipe_digests)<=1,"native recipes differ across performance rounds")
    counts=Counter((r["mode"],r["order"]) for r in pairs)
    expected={"no-profiler":{"ab":3,"ba":3},"l0":{"ab":2,"ba":2},
              "l1":{"ab":int(comparison in ("fixed-cache","default-cache")),"ba":0}}
    coverage={m:{o:dict(observed=counts[m,o],planned=n) for o,n in by_order.items()} for m,by_order in expected.items()}
    return dict(comparison=comparison,baseline=cfg["baseline"],candidate=cfg["candidate"],manifest_sha256=digest(manifest),
                branch_commit=cfg.get("branch_commit"),branch_library_sha256=cfg.get("branch_library_sha256"),
                schedule_sha256=schedule_hash,settings=settings,coverage=coverage,
                design_complete=all(counts[m,o]>=n for m,by_order in expected.items() for o,n in by_order.items()),
                all_have_host_receipts=bool(pairs) and all(r["proof"]["status"]=="host_receipt" for r in pairs),
                pairs=pairs,summary={o:aggregate(pairs,o) for o in ("all",*ORDERS)})


def display(v):
    return "—" if v is None else f"{v:.3f}"


def render(data, out):
    summary=data["summary"]
    lines=[f"# {data['comparison']} 动态 HSTU 分支复测","",f"A 为 {data['baseline']}，B 为 {data['candidate']}。",
           "",f"设计窗口{'已齐' if data['design_complete'] else '未齐'}。下表计数单位为独立 A/B 配对，括号内为计划数。","",
           "|模式|AB|BA|","|---|---:|---:|"]
    for m,by in data["coverage"].items():
        lines.append("|"+m+"|"+"|".join(f"{by[o]['observed']} ({by[o]['planned']})" for o in ORDERS)+"|")
    lines+=["","原始同步窗口为 10 步预热、50 步测量。配对差为 A−B，正值表示候选步时较短。样本/s、token/s 均按实际 shape 与整窗秒数折算。","",
            "|配对|顺序|A / B ms/步|节省 ms/步|步时降幅 %|A / B 样本/s|A / B token/s|","|---|---|---:|---:|---:|---:|---:|"]
    for r in data["pairs"]:
        if r["mode"]=="no-profiler":
            pair=lambda k:f"{r['A'][k]:.3f} / {r['B'][k]:.3f}"
            lines.append(f"|{r['id']}|{r['order'].upper()}|{pair('ms')}|{r['delta_ms']:.3f}|{r['gain_pct']:.3f}|{pair('samples_per_s')}|{pair('tokens_per_s')}|")
    lines+=["","|分组|配对数|步时改善次数|配对差中位 ms|最小 / 最大 ms|样本标准差 ms|配对降幅中位 %|",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for o,s in summary.items():
        st=s["paired_ms"]
        if st["n"]:
            lines.append(f"|{o.upper()}|{st['n']}|{s['positive_e2e_pairs']}|{st['median']:.3f}|{st['min']:.3f} / {st['max']:.3f}|{display(st['sample_sd'])}|{s['paired_e2e_gain_pct']['median']:.3f}|")
    lines+=["","Level0 每腿仅统计第 4–6 步。组内先合并 Free 与 Stage 时长，再计算占比。八列按 ALL、AB、BA 的次序分别展示可用分组，E2E 取该组各腿步时中位数，八列降幅从对应两列推导。",""]
    for order,s in summary.items():
        if "eight" not in s:
            continue
        lines += [f"**{order.upper()}**","","|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|",
                  "|"+"|".join(display(s["eight"][k])+(" ms" if k.endswith("_ms") else "%") for k in EIGHT)+"|",""]
    lines += ["|Level0 配对|顺序 / 腿|第 4–6 步实际 shape|Free ms/步|Stage ms/步|Computing ms/步|ΣFree/ΣStage %|",
              "|---|---|---|---:|---:|---:|---:|"]
    for r in data["pairs"]:
        if r["mode"]=="l0":
            for arm in r["order"].upper():
                v=r[arm];shapes="、".join(str(s["shape"]) for s in v["steps"])
                lines.append(f"|{r['id']}|{r['order'].upper()} / {arm}|{shapes}|{v['free_ms']:.3f}|{v['stage_ms']:.3f}|{v['computing_ms']:.3f}|{v['free_pct']:.3f}|")
    l1=[r for r in data["pairs"] if r["mode"]=="l1"]
    if l1:
        lines += ["","Level1 各步保留请求、实际物理流、控制任务及设备工作序列，并提供各算子的 producer、队列等待、consumer、ACL 提交时延与前邻设备空档分解。",""]
        for r in l1:
            labels=[f"#{s} {v['shape']} 首见 #{v['first_seen']}，同流工作顺序{'相同' if v['same_physical_streams_and_work_order'] else '有差异'}"
                    for s,v in r["level1_compare"].items()]
            lines.append(f"- {r['id']}，"+"；".join(labels)+f"。[逐算子与流对照]({r['level1_file']})")
        lines+=["","|配对 / 腿 / 步|实际 shape|已关联 enqueue/dequeue|逐流实体工作数|ACL kernel / memcpy 提交数|producer 准备|有符号队列等待|consumer 服务|单个 ACL kernel 提交|",
                "|---|---|---:|---|---:|---:|---:|---:|---:|"]
        for r in l1:
            detail=read(out/r["level1_file"])
            for arm in r["order"].upper():
                for step,rec in detail[arm]["steps"].items():
                    streams=", ".join(f"{s}={x['work_n']}" for s,x in rec["physical"].items())
                    acl=rec["acl_api"]
                    kernels=sum(n for k,n in acl.items() if "LaunchKernel" in k)
                    copies=sum(n for k,n in acl.items() if "Memcpy" in k)
                    all_lat=rec["all"]["latency"]
                    timings=[]
                    for key in ("aten_to_enqueue_ns","signed_queue_wait_ns","dequeue_ns"):
                        st=all_lat[key]
                        timings.append(f"{st['p50_us']:.2f} / {st['p99_us']:.2f}" if st["n"] else "—")
                    la=rec["exposure"]["single_acl_launch_us"]
                    timings.append(f"{la['p50']:.2f} / {la['p99']:.2f}" if la else "—")
                    lines.append("|"+"|".join([f"{r['id']} / {arm} / {step}",str(rec["shape"]),str(rec["all"]["calls"]),
                                                 streams,f"{kernels} / {copies}",*timings])+"|")
        lines+=["","末四列为 p50 / p99，单位 µs。producer 准备从能唯一定位的外层 ATen 起点到 enqueue 起点。队列等待从 enqueue 返回到 dequeue 开始，负值表示两段交叠。consumer 服务包含 ACL 调用。完整 JSON 单列 workspace、各阶段样本数、长尾及其与设备空档的名义交集。首次实际 shape 指该 shape 在本次训练序列中的首次出现，不能据此断言所有底层算子首次编译。"]
        lines+=["","|配对 / 步|op|两腿共同 ATen 序号|两腿均能关联物理工作|同流且任务数相同|同流且任务类型、名称序列相同|",
                "|---|---|---:|---:|---:|---:|"]
        for pair in l1:
            for step,comp in pair["level1_compare"].items():
                for op,rec in comp["embedding_same_request"].items():
                    lines.append(f"|{pair['id']} / {step}|{op}|{rec['shared']}|{rec.get('both_have_device_links',0)}|"
                                 f"{rec.get('same_streams_and_task_counts',0)}|{rec.get('same_stream_task_types_and_names',0)}|")
    lines += ["","[未舍入窗口和配对统计](metrics.json)"]
    return "\n".join(lines)+"\n"


def window_csv(data):
    stream=io.StringIO()
    fields=("comparison","id","mode","order","arm","metrics_sha256","ms_step","samples","tokens","elapsed_s",
            "samples_per_s","tokens_per_s","paired_ms_A_minus_B","paired_step_gain_pct","profile_step","batch","length",
            "free_ms","stage_ms","computing_ms","three_step_free_pct","paired_free_ratio_gain_pct")
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator="\n")
    writer.writeheader()
    for r in data["pairs"]:
        for arm in r["order"].upper():
            row=dict(comparison=data["comparison"],id=r["id"],mode=r["mode"],order=r["order"],arm=arm,
                     metrics_sha256=r["metrics_sha256"][arm])
            if r["mode"]=="no-profiler":
                s=r[arm]
                writer.writerow(dict(row,ms_step=s["ms"],samples=s["samples"],tokens=s["tokens"],elapsed_s=s["seconds"],
                                     samples_per_s=s["samples_per_s"],tokens_per_s=s["tokens_per_s"],
                                     paired_ms_A_minus_B=r["delta_ms"],paired_step_gain_pct=r["gain_pct"]))
            elif r["mode"]=="l0":
                for v in r[arm]["steps"]:
                    writer.writerow(dict(row,profile_step=v["step"],batch=v["shape"][0],length=v["shape"][1],
                        free_ms=v["Free"],stage_ms=v["Stage"],computing_ms=v["Computing"],
                        three_step_free_pct=r[arm]["free_pct"],paired_free_ratio_gain_pct=r["free_ratio_gain_pct"]))
    return stream.getvalue()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--study",type=Path,default=Path(__file__).resolve().parents[2])
    args=parser.parse_args()
    out=local_path(Path.cwd(),args.out)
    out.mkdir(parents=True,exist_ok=True)
    data=run(args.manifest.resolve(),out,args.study.resolve())
    data["analysis_sha256"]=digest(Path(__file__))
    write(out/"metrics.json",json_text(data))
    write(out/"windows.csv",window_csv(data))
    write(out/"REPORT.md",render(data,out))
    print(f"{data['comparison']}: {len(data['pairs'])} completed pairs, design_complete={data['design_complete']}")


if __name__=="__main__":
    main()

