# v2.10 动态 HSTU 分支复测的度量与证据

`measure.py` 在本地读取原 runner 的两腿产物。候选的提交号和原生库 SHA 取自构建收据；每个正式配对固定 10 步预热、50 步同步测量。无 profiler 首尾同步，逐步 shape 与原始 `metrics.json`、`--shape-schedule` 哈希一致。吞吐使用这 50 步的样本与 token 总和。在当前序列中分母为 136 个样本和 2,994 个 token。

| 比较 | A | B | 计划 |
|---|---|---|---|
| fixed-cache | 同一新包，全关，65,536 项缓存 | 同包，全开，65,536 项缓存 | 无 profiler 3 AB + 3 BA；L0 2 AB + 2 BA；L1 1 AB |
| stock-to-full | 原镜像库，默认缓存 | 新包，全开，65,536 项缓存 | 无 profiler 3 AB + 3 BA；L0 2 AB + 2 BA |
| default-cache | 同一新包，全关，默认缓存 | 同包，全开，默认缓存 | 无 profiler 3 AB + 3 BA；L0 2 AB + 2 BA；L1 1 AB |

另有每个比较各一组八步 A/B 语义审计。审计独立归档，不计入正式窗口。每个配对有独立 ID，`01-A/02-B` 或 `01-B/02-A` 确定执行顺序。分析核对原 `plan.json`、`status.json`、`train.exit`、两腿实际 shape、启动训练参数、两腿载入的 .so、实际六个开关和原 worker PID；已结束训练的 NPU 8 在两个边界都应是板 4/芯片 0、健康、无其他上下文。将原机每次全卡采样与 PID 对齐，只有窗口内所有快照有效、无外部进程，且至少一次看到目标 worker，才计入正式统计。[度量与校验](measure.py)

## 本地输入和复算

约定数据在 `branch-stability-v210/data`，其中 `expected-schedule.json` 是所有轮次的调度计划，`dynamic_shapes_npu8.json` 是逐步实际形状序列。每个比较输入一份 `data/manifests/<comparison>.json`，其中 `raw`、`receipt`、`chip_window`、`schedule` 均相对于该 manifest，允许仅列已经验证的配对。下面为单轮字段示例，原生库 SHA 和原始 run ID 使用对应收据中的实际值。

```json
{
  "comparison": "fixed-cache",
  "baseline": "新包，全关，65,536 项缓存",
  "candidate": "新包，全开，65,536 项缓存",
  "schedule": "../dynamic_shapes_npu8.json",
  "pairs": [
    {
      "id": "branch-v210-fixed-cache-no-profiler-01-ab",
      "mode": "no-profiler",
      "order": "ab",
      "raw": "../raw/<runner 生成的 run ID>",
      "receipt": "../pairs/branch-v210-fixed-cache-no-profiler-01-ab",
      "chip_window": "../chips/branch-v210-fixed-cache-no-profiler-01-ab.jsonl"
    }
  ]
}
```

在 `branch-stability-v210` 下执行下面的命令。分析复用上一级现有的 `runner/paired_metrics.py`、`queue/analyze_pair.py`、`queue/exposure.py` 和 `followup70/producer-sched/analyze_dynamic_trace.py`。两份完整 Level1 仅在原文件和解析器哈希有变化时重建。

```sh
python -B analysis/measure.py data/manifests/fixed-cache.json --out reports/fixed-cache
python -B analysis/collect.py --local-only
```

`reports/<comparison>/windows.csv` 给出每个原始同步窗或每个 L0 步的未舍入数据；`metrics.json` 包含每轮对照、按顺序分组与所有轮次的统计；`REPORT.md` 汇总八列、逐轮吞吐、Level0 和 Level1。Level1 的 `pair.json` 包含两腿逐步、逐算子、逐流的请求数与设备任务名。相邻 `linked/` 中的 `queue_calls.csv.gz`、`work_tasks.csv.gz`、`device_gaps.csv.gz` 分别保留单请求阶段、物理任务与前置全设备空档。未能唯一关联的工作显式列为 `<unlinked>`。分别比较物理流编号一致和允许重编号时的工作顺序，区分 control 任务与实际 kernel、DMA、通信。

## 口径与稳定性

- 步时为同步窗总秒数 × 1,000 / 50，配对差为 A−B，配对降幅为 100×(1−B/A)。分别报告全部、AB 和 BA 的中位数、最小/最大、样本标准差、MAD 及正值次数。根据各轮符号与两个顺序报告观测稳定性；没有预设达标百分比或统计显著性阈值。
- Level0 仅读第 4–6 步，显示原 shape、Free、Stage、Computing 的 ms/步。每条腿先求 ΣFree/ΣStage，再比较占比的相对下降。八列表中每个顺序的 E2E 取该组各腿步时中位数，Free/Stage 汇集该组独立的 L0 步重新计算，绝不把一个 L0 窗口重复分配给多个无 profiler 轮次。另列真正成对的每轮差及其中位数。
- Level1 仅解释实际采到的第 4 和 5 步。单次提交比较核对同一步 ATen 序列、请求量、ACL launch/memcpy 数，以及每条物理流上的实际任务数和顺序。未满足等工作量条件时，须把设备提交数变化作为单独机制报告。各阶段 p50/p95/p99/max 包含 producer 的 ATen 起点到 enqueue、workspace、enqueue 调用、队列等待、consumer dequeue 与 ACL 调用。阶段嵌套，不相加。空档中的阶段划分采取原 trace 的名义校时，区间重合不单独证明因果。首次 shape 只指 `[B,L]` 在本次训练首次出现。
- 时钟、卡况与共享主机负载作为原始收据保留。`data/evidence-index.json` 给出每个 run 和 trace 的远端原路径、解析路径与 SHA-256，`data/occupancy.json` 逐腿列出原 PID、芯片采样数和对应负载。调度器自身因停止、重试而移出的失败尝试不进入正式配对。

## 回收端

`collect.py --watch` 通过已校验主机公钥的 SSH 登录。每轮只有 `slot-verified.json` 已写成才进行下载。远端只在共享归档卷写收据包，以 nice 19、idle I/O 读取原实验结果，无设备操作。小窗持续回收，Level1 原 trace 在这次独立采样和卡况检查结束后下载。本地逐项验 SHA、写入 `data/pairs`、`data/raw`、`data/chips` 与 `data/sources`。已回收的不可变轮次可在重连后复用。所有统计和 trace 解析在本地执行。[远端收据打包](pack_receipts.py) · [回收与聚合](collect.py)


## 完整归档的离线复验

两组 Level1 的 linked CSV 生成后，`python -B analysis/side_tables.py` 单独复算各步首张及其余 1,023 张侧表，在 `reports/side-tables.md` 和 `reports/side-tables.json` 保存 per-request 与 per-ACL 数据。发布程序检查两份归因文件。

正式 35 组收齐后，运行 `python -B analysis/collect.py --local-only` 读取所有本地收据包，逐成员校验 SHA 并恢复原目录。图使用 `python -B analysis/plot_pairs.py` 从 18 个同步配对和 12 个 Level0 配对重算，需要 Matplotlib 与中文字体。最后运行 `python -B analysis/publish.py`。仅在三组窗口、35 组占卡收据、三组八步审计与配对图均齐全时，生成正式总表、两份可移植附件和发布清单。将两份附件展开至同一个 `infra-only-20261128` 目录，可在其中的 `branch-stability-v210` 重复离线校验与发布流程。归档已附成图与 linked CSV，读取结果无需 Matplotlib、SSH 或 NPU。
