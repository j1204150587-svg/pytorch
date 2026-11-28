"""Produce a standalone stability report and portable, checksummed evidence after receipt recovery."""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import tarfile
from collections import Counter

from measure import COMPARISONS, EIGHT, digest, display, json_text, local_path, read, write

HERE=Path(__file__).resolve().parent


def verdict(values):
    if all(x>0 for x in values):return "各轮均改善"
    if all(x<0 for x in values):return "各轮均回退"
    if all(x==0 for x in values):return "各轮均为零"
    if min(values)<0<max(values):return "配对方向反转"
    return "观测同时含零和非零"


def evidence_archive(path,contents):
    with tempfile.NamedTemporaryFile(delete=False,dir=path.parent) as f:
        temp=Path(f.name)
    try:
        with temp.open("wb") as raw, gzip.GzipFile(fileobj=raw,mode="wb",filename="",mtime=0,compresslevel=1) as gz:
            with tarfile.open(fileobj=gz,mode="w") as tar:
                index={}
                for name,source in sorted(contents.items()):
                    entry=tarfile.TarInfo(name);entry.size=source.stat().st_size;entry.mode=0o644
                    with source.open("rb") as fp:tar.addfile(entry,fp)
                    index[name]=dict(bytes=entry.size,sha256=digest(source))
                data=json_text(index).encode()
                entry=tarfile.TarInfo("MANIFEST.json");entry.size=len(data);tar.addfile(entry,io.BytesIO(data))
        with tarfile.open(temp,"r:gz") as tar:
            expected=json.load(tar.extractfile("MANIFEST.json"))
            assert len(expected)+1==len(tar.getmembers())
            for name,record in expected.items():
                with tar.extractfile(name) as fp:
                    assert hashlib.file_digest(fp,"sha256").hexdigest()==record["sha256"]
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def kernel_delta(before, after):
    def counts(work):
        return Counter({(stream, name): count for stream, names in work["by_stream"].items()
                        for name, count in names.items()})
    a, b = counts(before), counts(after)
    def format_changes(changes):
        return ", ".join(f"`{name}` ×{n} (流 {stream})" for (stream,name),n in sorted(changes.items())) or "无"
    return format_changes(a-b), format_changes(b-a)


def targets(detail):
    rows = []
    links = []
    gaps = []
    for step, comp in detail["compare"].items():
        for op, matched in comp["embedding_same_request"].items():
            arms = [detail[arm]["steps"][step] for arm in "AB"]
            work = [leg["request_and_device_work"][op] for leg in arms]
            removed, added = kernel_delta(*work)
            links.append(f"|{step}|{op}|{matched['shared']}|{matched.get('both_have_device_links',0)}|"
                         f"{matched.get('same_streams_and_task_counts',0)}|{matched.get('same_stream_task_types_and_names',0)}|"
                         f"{work[0]['acl_submissions']}→{work[1]['acl_submissions']}|{removed}|{added}|")
            for arm, leg in zip("AB",arms):
                group = leg["ops"][op]
                fields = []
                for key in ("aten_to_enqueue_ns", "workspace_ns", "signed_queue_wait_ns", "dequeue_ns", "acl_api_sum_ns"):
                    st = group["latency"][key]
                    fields.append(f"{st['p50_us']:.2f} / {st['p99_us']:.2f} ({st['n']})" if st["n"] else "— (0)")
                rows.append("|"+"|".join([step,op,arm,*fields,str(group["long_wait_gt100us"])])+"|")
                gap = group["next_gap_ms"]
                gaps.append("|"+"|".join([step,op,arm]+[f"{gap.get(k,0):.3f}" for k in
                            ("gap_ns","workspace_ns","visible_queue_wait_ns","consumer_before_api_ns","acl_submission_api_ns","after_api_ns")])+"|")
    return ["|步|算子|腿|producer 准备|workspace|有符号队列等待|consumer 服务|每请求 ACL 合计|队列等待 >100 µs 次数|",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",*rows,"",
            "各格为 p50 / p99（样本数），单位 µs。producer 准备从已对齐的 ATen 起点算至 enqueue 开始，通常包含 workspace。workspace 计数仅覆盖具有对应打点的请求。consumer 服务包含 ACL，每请求 ACL 合计可覆盖多次 API。","",
            "|步|算子|共同请求|双腿有设备连线|同流且任务数一致|同流、类型、名称、顺序均一致|每请求汇总的 ACL 调用数 A→B|设备工作减少|设备工作增加|",
            "|---:|---|---:|---:|---:|---:|---:|---|---|",*links,"",
            "设备工作按物理流和任务名分别计数。对齐键为调用线程角色、ATen 名称和该线程内序号。相同提交数可包含 kernel 名称替换。","",
            "|步|算子|腿|前邻设备空档总 ms|workspace 相交 ms|可见队列等待相交 ms|consumer 至 API 相交 ms|ACL 提交相交 ms|API 后相交 ms|",
            "|---:|---|---|---:|---:|---:|---:|---:|---:|",*gaps,"",
            "按下一项设备工作所属请求归集前邻空档，上表列出重点区间，全部阶段及每个空档的时间戳见 linked CSV。相交时长用于定位暴露开销，无法单凭交叠判定因果。",""]


def markdown(all_results,occupancy,audits,reports):
    lines=["# v2.10 动态 HSTU 分支复测","",
           f"分支提交 `{all_results[0]['branch_commit']}`，轮次中实际加载的分支原生库 SHA-256 为 `{all_results[0]['branch_library_sha256']}`。"
           "三组分别测量大缓存内的代码开关、原镜像到完整配置的端点变化，以及默认缓存内的代码开关。每组包含 3 个 AB、3 个 BA 的无 profiler 配对，另有 2 个 AB、2 个 BA 的 Level0 配对。","",
           "训练沿用动态 eager、1,024 张侧表、单 rank、梯度累积 1、BF16 ZeRO-2、01 分块 overflow 和 Task Queue Mode 1。"
           "60 个合成 shape 的 batch size 为 1–4，序列长度为 12–32，每步直接生成实际尺寸。每个无 profiler 窗口预热 10 步、首尾同步测量 50 步，"
           "实际计 136 个样本、2,994 个 token。Free 来自 Level0 的第 4–6 步，依次为 [1,19]、[2,14]、[2,22]。","",
           "本轮构建含三个补丁。反向与前向分别处理 embedding 的 indices 描述符及缓存复用，Task Queue 补丁改为在满队列有登记等待者时唤醒 producer。"
           "[反向 embedding 补丁](https://github.com/j1204150587-svg/pytorch/commit/2cd525d5563cea44fb17596830c3a2367a99dff9) · "
           "[前向 embedding 补丁](https://github.com/j1204150587-svg/pytorch/commit/645b89a69c34509e28775b82a5bc478fb33ca556) · "
           "[Task Queue 唤醒补丁](https://github.com/j1204150587-svg/pytorch/commit/e3cda6399cb194c813530222478ea018f5aca5c9)","",
           "三组端点、重建 wheel 与映射库、默认缓存值及显式 65,536 条目的核对、训练语义和整卡收尾状态均有独立记录。"
           "[实验设计](../实验设计.md) · [分支构建](../build/RESULTS.md) · [缓存核对](../build/CACHE_LIMIT.md) · "
           "[实际激活与非有限梯度跳步](../semantics/README.md) · [最终占卡状态](../final-slot.json)","",
           "|比较|六轮步时降幅中位数 %|六轮步时降幅范围 %|正向轮数|AB / BA 配对降幅中位数 %|Free 占比改善轮数|结论|",
           "|---|---:|---:|---:|---:|---:|---|"]
    for report in all_results:
        s=report["summary"]["all"]
        pct=s["paired_e2e_gain_pct"]
        e2e=[r["gain_pct"] for r in report["pairs"] if r["mode"]=="no-profiler"]
        free=[r["free_ratio_gain_pct"] for r in report["pairs"] if r["mode"]=="l0"]
        order=" / ".join(display(report["summary"][o]["paired_e2e_gain_pct"]["median"]) for o in ("ab","ba"))
        lines.append(f"|[{report['comparison']}](../reports/{report['comparison']}/REPORT.md)|{pct['median']:.3f}|"
                     f"{pct['min']:.3f} / {pct['max']:.3f}|{sum(x>0 for x in e2e)}/6|{order}|{sum(x>0 for x in free)}/4|"
                     f"步时{verdict(e2e)}，Free 占比{verdict(free)}|")
    lines+=["","端到端降幅按同一配对的 A→B 计算。八列中各腿步时取六轮中位数，Free 占比使用该腿 4 个配对共 12 个实测步的 ΣFree/ΣStage。"
            "独立 E2E 与 Level0 的聚合分母在各组报告中逐项列出。","",
            "![三个比较的同步窗口与 Level0 各轮配对降幅](../reports/paired-runs.png)",""]
    detailed = []
    for report in all_results:
        cols=report["summary"]["all"]["eight"]
        s=report["summary"]["all"];a=s["legs"]["A"];b=s["legs"]["B"]
        lines += [f"### {report['comparison']}","",f"A 为{report['baseline']}，B 为{report['candidate']}。","",
                  "|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|",
                  "|"+"|".join(display(cols[k])+(" ms" if k.endswith("_ms") else "%") for k in EIGHT)+"|","",
                  f"样本/s 中位数 {a['samples_per_s']['median']:.3f} → {b['samples_per_s']['median']:.3f}，token/s 中位数 "
                  f"{a['tokens_per_s']['median']:.3f} → {b['tokens_per_s']['median']:.3f}。每个同步窗口均按真实工作量计数。",""]
        free=s["paired_free_ratio_gain_pct"]
        lines += [f"四次 Free 占比配对相对降幅为 {free['min']:.3f}% 至 {free['max']:.3f}%，"
                  f"中位数 {free['median']:.3f}%。A/B 的 Stage 为 {a['stage_ms']:.3f}/{b['stage_ms']:.3f} ms/步，"
                  f"Computing 为 {a['computing_ms']:.3f}/{b['computing_ms']:.3f} ms/步。",""]
        durations=[r for r in report["pairs"] if r["mode"]=="no-profiler"]
        limits=sorted(durations,key=lambda r:r["gain_pct"])
        lines+=["|步时降幅边界|配对轮次 / 顺序|A→B ms/步|配对降幅|A/B 宿主 1 分钟负载中位数|A/B 全机 CPU 忙碌率|",
                "|---|---|---:|---:|---:|---:|"]
        for label,row in zip(("最小","最大"),(limits[0],limits[-1])):
            loads=[occupancy[row["id"]][arm]["host_load"] for arm in "AB"]
            busy=" / ".join(display(load.get("whole_machine_busy_pct")) for load in loads)
            one=" / ".join(display(load.get("median_1m")) for load in loads)
            lines.append(f"|{label}|`{row['id']}` / {row['order'].upper()}|{row['A']['ms']:.3f}→{row['B']['ms']:.3f}|"
                         f"{row['gain_pct']:.3f}%|{one}|{busy} %|")
        lines+=["","宿主负载来自与训练进程生命周期重叠的周期采样，包含启动和回收。训练线程所在核心的排队时长没有独立计数。",""]
    lines+=["## 首见形状的 Level1 对照","",
            "|比较 / 步|实际 shape|实际工作 A→B|已关联队列请求 A→B|队列请求的计数差|同线程 ATen 顺序相同|相同物理流与任务序列|ACL 调用数相同|",
            "|---|---|---:|---:|---|---|---|---|"]
    for report in all_results:
        for pair in report["pairs"]:
            if pair["mode"]!="l1":continue
            data=read(reports/report["comparison"]/pair["level1_file"])
            detailed += [f"### {report['comparison']} 的嵌入前反向","","具体函数为 `aclnnEmbedding`、`aclnnEmbeddingDenseBackward`。","",*targets(data)]
            for step,comp in data["compare"].items():
                ra,rb=(data[arm]["steps"][step] for arm in "AB")
                counts=[sum(r["work_n"] for r in m["physical"].values()) for m in (ra,rb)]
                queues=[r["enqueue_counts"] for r in (ra,rb)]
                diff=", ".join(f"`{name}` {queues[0].get(name,0)}→{queues[1].get(name,0)}" for name in
                               sorted(queues[0].keys()|queues[1].keys()) if queues[0].get(name,0)!=queues[1].get(name,0)) or "相同"
                lines.append(f"|{report['comparison']} / {step}|{ra['shape']}|{counts[0]}→{counts[1]}|{ra['all']['calls']}→{rb['all']['calls']}|"
                             f"{diff}|{'是' if comp['same_aten_per_thread_order'] else '否'}|"
                             f"{'是' if comp['same_physical_streams_and_work_order'] else '否'}|{'是' if comp['same_acl_counts'] else '否'}|")
    lines += ["","两种缓存配置都观察到每步 17,220 或 17,221 项实际设备工作，A/B 各类 ACL 提交数相等。嵌入前反向的 Host 参数准备和 consumer 服务时长下降。"
              "每步每个目标算子共有 1,027 个按序号对齐的请求，其中 1,024 个改变了具体设备任务名，流与任务数一致。"
              "这组整包配对支持 Host 请求缩短并伴随 kernel 替换，无法将整包步时收益分配给单个补丁，也无法把全部变化归给单次同名 ACL launch。","",
              "两个 Level1 录制只覆盖首见形状 [1,19]、[2,14]。60 个合成形状各出现一次，本组没有同一实际 shape 再次执行的对照。"
              "首张与同一步其余 1,023 张侧表按顺序、物理流以及实际 dtype 和输入 shape 单列。"
              "[首张及后续侧表逐请求归因](../reports/side-tables.md) · [未舍入数据](../reports/side-tables.json)","",*detailed]
    all_legs=[leg for pair in occupancy.values() for leg in pair.values()]
    gates=[info["training"] for info in audits.values()]
    gate=read(reports.parent/"semantics/activation.json")
    act=gate["original_training_audit"]["arms"]
    fwd=act["B"]["counts"]["forward"]; bwd=act["B"]["counts"]["backward"]
    q_a=act["A"]["queue"];q_b=act["B"]["queue"]
    overflow=gate["probes"]["overflow-B"]["checks"]
    lines+=["","每次实际请求的 producer、队列等待、consumer、ACL 调用分位数和对应设备空档，都可在各组 REPORT 以及逐笔 linked CSV 中核对。物理任务序列和启动 API 未对齐时，不能据此声称单次提交提速。","",
            f"{len(occupancy)} 个配对的 {len(all_legs)} 条腿逐一核对卡映射、两端无残留、真实 PID 与整卡采样。每腿采样 "
            f"{min(v['samples'] for v in all_legs)}–{max(v['samples'] for v in all_legs)} 次，均见到目标 worker，"
            "异常/外部进程快照为 0。共享宿主负载随时间变化；本表记录轮次波动，不给出未经配对验证的固定收益。","",
            f"三种配置的八步训练审计均通过。原始结果中 norm 差最大为 {max(g['norm_max_delta'] for g in gates):.6g}，"
            f"loss 最大绝对差为 {max(g['loss_max_delta'] for g in gates):.6g}，各腿梯度、训练权重及 AdamW 状态通过原审计。","",
            f"大缓存配置的八步审计中，B 前向命中 {fwd[0]:,} 次、回退 {fwd[1]} 次，反向命中 {bwd[0]:,} 次、回退 {bwd[1]} 次。"
            f"两腿消费队列项为 {q_a['drained']:,} / {q_b['drained']:,}，未观察到登记 producer 等待者时的 eventfd 写由 "
            f"{q_a['writes_without_observed_waiter']:,} 次降至 {q_b['writes_without_observed_waiter']:,} 次。独立 ZeRO-2 探针覆盖 "
            f"{overflow['finite_updates']} 次有限更新和 {overflow['nonfinite_skips']} 次非有限梯度跳步，A/B 的梯度、权重、AdamW 状态逐步一致。"
            "[实际计数和微门收据](../semantics/activation.json)","",
            "每个配对的源目录、原 trace 路径、文件大小及 SHA-256 在证据索引中。正式采样期间的全机负载快照、worker 映射、原始命令、shape 和分支审计收据收录于原始包。"
            "[逐配对源与 trace 哈希](../data/evidence-index.json) · [各腿卡况](../data/occupancy.json) · [八步审计](../data/audits.json) · "
            "[原始收据与轨道](原始收据与轨道.tgz) · [汇总与复算程序](汇总与复算程序.tgz)",""]
    return "\n".join(lines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=HERE.parent)
    args=p.parse_args()
    root=local_path(Path.cwd(),args.root)
    study=root.parent
    data=root/"data";reports=root/"reports";out=root/"publication"
    expected=read(data/"expected-schedule.json");source=read(data/"evidence-index.json")
    assert len(expected)==35 and {r["id"] for r in source}=={r["id"] for r in expected} and len(source)==35
    results=[read(reports/c/"metrics.json") for c in COMPARISONS]
    assert all(r["design_complete"] and r["all_have_host_receipts"] for r in results)
    assert all(r["analysis_sha256"]==digest(HERE/"measure.py") for r in results)
    occupancy=read(data/"occupancy.json");audits=read(data/"audits.json")
    assert len(occupancy)==35 and len(audits)==3
    build=read(data/"build-ready.json")
    designs={row["id"]:row for row in expected}
    shape_file=data/"dynamic_shapes_npu8.json"
    schedule=read(shape_file)
    assert len(set(map(tuple,schedule)))==len(schedule)==60
    assert (min(s[0] for s in schedule),max(s[0] for s in schedule),min(s[1] for s in schedule),max(s[1] for s in schedule))==(1,4,12,32)
    cohort_configs=set()
    for receipt in source:
        config=data/"raw"/Path(receipt["raw"]).name/"workspace/configs"
        model=config/"model_many_tables.json"
        ds=config/"deepspeed_zero2_bf16_overflow_on.json"
        assert digest(config/shape_file.name)==digest(shape_file)
        assert read(model)["num_tables"]==1024
        deepspeed=read(ds)
        assert deepspeed["gradient_accumulation_steps"]==1 and deepspeed["bf16"]["enabled"] and deepspeed["zero_optimization"]["stage"]==2
        cohort_configs.add((digest(model),digest(ds)))
    assert len(cohort_configs)==1
    verified=[(pair["id"],pair["proof"]) for report in results for pair in report["pairs"]]
    verified.extend((identifier,item["worker_proof"]) for identifier,item in audits.items())
    assert len(verified)==35 and {identifier for identifier,_ in verified}==set(designs)==set(occupancy)
    for identifier,proof in verified:
        for arm,leg in proof["legs"].items():
            native=[value for path,value in leg["library_digests"].items() if path.endswith("/libtorch_npu.so")]
            stock=designs[identifier]["comparison"]=="stock-to-full" and arm=="A"
            assert native==[build["stock_elf_sha256"] if stock else build["sha256"]],(identifier,arm)
    with (reports/"paired-runs.png").open("rb") as picture:
        assert picture.read(8)==bytes.fromhex("89504e470d0a1a0a")
    assert (reports/"side-tables.md").stat().st_size>0 and read(reports/"side-tables.json")
    out.mkdir(exist_ok=True)
    write(out/"分支稳定性结果.md",markdown(results,occupancy,audits,reports))
    take={}
    for name in ("expected-schedule.json","dynamic_shapes_npu8.json","build-ready.json","occupancy.json","audits.json","evidence-index.json"):
        take["branch-stability-v210/data/"+name]=data/name
    for row in source:
        packet=data/"packets"/(row["id"]+".tgz")
        assert digest(packet)==row["packet"]["sha256"]
        take["branch-stability-v210/data/packets/"+packet.name]=packet
        rec=packet.with_suffix(".json")
        take["branch-stability-v210/data/packets/"+rec.name]=rec
    evidence_archive(out/"原始收据与轨道.tgz",take)
    derived={}
    for directory in (root/"analysis",data/"manifests",reports,root/"build",root/"semantics",root/"host"):
        for f in directory.rglob("*"):
            if f.is_file() and "input" not in f.relative_to(directory).parts and "__pycache__" not in f.parts:
                derived["branch-stability-v210/"+f.relative_to(root).as_posix()]=f
    for rel in ("runner/paired_metrics.py","queue/analyze_pair.py","queue/exposure.py","followup70/producer-sched/analyze_dynamic_trace.py"):
        derived[rel]=study/rel
    for rel in ("实验设计.md","final-slot.json"):
        derived["branch-stability-v210/"+rel]=root/rel
    derived["branch-stability-v210/publication/分支稳定性结果.md"]=out/"分支稳定性结果.md"
    evidence_archive(out/"汇总与复算程序.tgz",derived)
    manifest={f.name:dict(bytes=f.stat().st_size,sha256=digest(f)) for f in
              [out/"分支稳定性结果.md",out/"原始收据与轨道.tgz",out/"汇总与复算程序.tgz"]}
    write(out/"publish-manifest.json",json_text(manifest))
    print(json_text(manifest))

if __name__=="__main__":
    main()

