# 动态 eager HSTU 的 NPU 8 实测台账

60 个合成 shape 上，分组侧表查找使无 profiler 步时在 A/B、B/A 分别缩短 73.27%、75.09%，绝对 Free 下降，设备计算缩短得更多，Free 占比随之上升。把同一 ZeRO flat 上连续的侧表交给单一梯度叶处理，两序步时再相对该项各自的基线缩短 16.51%、19.63%，候选 Free 占比分别为 91.11%、90.72%。两项所用基线不同。

## 负载与测量

逻辑 NPU 8 映射到板 4、芯片 0，容器内为 npu:0。实际输入逐步按 60 个互异的 [batch_size, seq_len] 生成，batch 1–4，序列长度 12–32。1,024 张侧表、1,052 个参数张量、单 rank、BF16 ZeRO-2、累积 1、seed 1234、Task Queue Mode 1，默认启用 01 分块 overflow。沿用宿主共享锁、容器和设备进程前后检查。[实际形状](working/evidence/grouped-side-v1-no-profiler-ab/workspace/configs/dynamic_shapes_npu8.json) · [前序动态基线和调用归因](../动态NPU8优化台账.md)

无 profiler 先预热 10 步，首尾同步测 50 步，合计 136 个样本、2,994 个 token。Level0 取第 4–6 步，实际为 [1,19]、[2,14]、[2,22]。Free 占比按三步 ΣFree/ΣStage 计算，八列的改善按 1−B/A 复算。双序表首行对应 A/B，次行对应 B/A。

独立的 01 stock 动态 eager 基线为 600.060 ms/步，Level0 Free 占比 92.738%。E01 两腿另共用 07 梯度打包、分块 norm、ACLNN executor cache 上限 65,536；该共同配置的独立同步窗口为 326.284 ms/步，Level0 Free 393.024、Stage 442.110、Computing 48.978 ms/步，Free 占比 88.90%。各候选的增量以各自配对的 A 计。[共同配置复算](reports/packed-norm-cache/paired_results.json) · [01 基线](../动态NPU8优化台账.md)

## 有实测收益的优化

### E01 1,024 张侧表的批量查找与有序 BF16 累加

**改动位置。** MultiTableHSTU.forward 的侧表循环改调 Torch dispatcher 的 eager 复合算子。批量 embedding、mean 后，Triton fold 按原表顺序逐次 BF16 舍入。动态网格使用实际 B×L。CatBackward 仍将梯度分给原 1,024 个参数。8 步训练门核对输入、梯度、权重、AdamW 状态，以及 NaN、+Inf、−Inf 注入后的跳步和恢复。[侧表实现](side_tables) · [实机逐位检查](working/diagnostics/side-micro-v1/micro.json) · [训练语义门](working/gates/grouped-side-v1)

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 283.913 ms/步 | 75.896 ms/步 | +73.27% | 417.892 ms/步 | 89.29% | 172.494 ms/步 | 96.59% | -8.17% |
| 326.112 ms/步 | 81.244 ms/步 | +75.09% | 364.029 ms/步 | 88.01% | 99.985 ms/步 | 94.23% | -7.06% |

AB 整步缩短 208.017 ms。样本/s 为 9.580 → 35.838，token/s 为 210.910 → 788.970。

BA 整步缩短 244.868 ms。样本/s 为 8.341 → 33.479，token/s 为 183.618 → 737.037。

[分步 Stage、Computing 和完整同步窗口](reports/grouped-side-v1/measurement.md) · [同轮未舍入数值](reports/grouped-side-v1/paired_results.json) · [实际启动配方](working/registries/grouped-side-v1/recipe.json)

### E02 将同一 ZeRO flat 上连续的侧表作为一个梯度叶

**改动位置。** SideSlab.forward 复用已验证的 ZeRO 权重分区，批量查表可以直接读取连续平坦视图。反向完成后，publish 将 dense 梯度复制到持久缓冲区，用预建的 1,024 个视图交回原梯度打包与优化器。每步校验参数身份、地址、梯度与 hook。两腿均已启用 E01、#09 分区提示、07、分块 norm 和缓存。本项改变梯度表示与设备工作量。[eager 实现](side_slab/side_slab.py) · [双序同一训练语义门](working/gates/side-slab-v1/gate-attempt-g4u9yvic/comparison.json)

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 62.068 ms/步 | 51.823 ms/步 | +16.51% | 79.808 ms/步 | 93.11% | 60.280 ms/步 | 91.11% | +2.15% |
| 61.708 ms/步 | 49.592 ms/步 | +19.63% | 78.195 ms/步 | 93.07% | 58.457 ms/步 | 90.72% | +2.52% |

AB 整步缩短 10.245 ms。样本/s 为 43.823 → 52.487，token/s 为 964.751 → 1155.475。

BA 整步缩短 12.116 ms。样本/s 为 44.079 → 54.848，token/s 为 970.384 → 1207.455。

[分步 Stage、Computing 和完整同步窗口](reports/side-slab-v1/measurement.md) · [同轮未舍入数值](reports/side-slab-v1/paired_results.json) · [实际启动配方](working/registries/side-slab-v1/recipe.json)

## 已完成的单序观测

### 已有 #09 在分组查表路径上的复测

在 E01 共同基线上只开启已存在的 ZeRO 分区提示，重用已验证的平坦梯度和权重。 本地回执已包含同轮正序的同步窗口与 Level0。

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 72.264 ms/步 | 66.639 ms/步 | +7.78% | 92.907 ms/步 | 94.01% | 92.641 ms/步 | 93.98% | +0.03% |

AB 整步缩短 5.624 ms。样本/s 为 37.640 → 40.817，token/s 为 828.632 → 898.567。

[分步 Stage、Computing 和完整同步窗口](reports/grouped-flat-hints-r2/measurement.md) · [同轮未舍入数值](reports/grouped-flat-hints-r2/paired_results.json) · [实际启动配方](working/registries/grouped-flat-hints-r2/recipe.json)

### 聚合梯度打包中的同 storage 连续区间

两腿共用 E01、E02、#09、07、分块 norm 和缓存。B 将相邻侧表梯度视图作为一段输入交给同一次 cat，保留优化器所见的独立视图。 本地回执已包含同轮正序的同步窗口与 Level0。

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 51.831 ms/步 | 42.738 ms/步 | +17.54% | 57.900 ms/步 | 90.75% | 45.021 ms/步 | 88.39% | +2.60% |

AB 整步缩短 9.092 ms。样本/s 为 52.479 → 63.643，token/s 为 1155.302 → 1401.084。

[分步 Stage、Computing 和完整同步窗口](reports/grouped-slab-pack/measurement.md) · [同轮未舍入数值](reports/grouped-slab-pack/paired_results.json) · [实际启动配方](working/registries/grouped-slab-pack/recipe.json)

### 在已打包 BF16 flat 上直接检查非有限值

两腿共用 E01、E02、#09、07、分块 norm 和缓存。B 在验证 1,052 份梯度布局后，用一次全 flat 的 isfinite 检查替代分块处理。 本地回执已包含同轮正序的同步窗口与 Level0。

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 55.964 ms/步 | 43.298 ms/步 | +22.63% | 65.077 ms/步 | 91.77% | 50.107 ms/步 | 91.30% | +0.51% |

AB 整步缩短 12.666 ms。样本/s 为 48.602 → 62.821，token/s 为 1069.965 → 1382.975。

[分步 Stage、Computing 和完整同步窗口](reports/flat-overflow-v1/measurement.md) · [同轮未舍入数值](reports/flat-overflow-v1/paired_results.json) · [实际启动配方](working/registries/flat-overflow-v1/recipe.json)

### 侧表梯度直接发布到最终分区

两腿共用 E01、E02、#09、梯度分区打包、flat overflow/norm 和缓存。B 在 ZeRO 的反向分区结尾直接发布侧表梯度，剩余 28 个参数仍按原依赖检查。 本地回执已包含同轮正序的同步窗口与 Level0。

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 28.652 ms/步 | 22.194 ms/步 | +22.54% | 31.336 ms/步 | 89.84% | 28.286 ms/步 | 88.82% | +1.13% |

AB 整步缩短 6.458 ms。样本/s 为 94.933 → 122.554，token/s 为 2089.916 → 2697.999。

[分步 Stage、Computing 和完整同步窗口](reports/direct-publish-v1/measurement.md) · [同轮未舍入数值](reports/direct-publish-v1/paired_results.json) · [实际启动配方](working/registries/direct-publish-v1/recipe.json)

## GitCode !42634 的真实触发范围

上游功能提交 b1c30ebd，PR head 为 2701f7b3。实现面向 Inductor 生成的 NPU Triton Python Wrapper，稳定 launcher 可以创建并复用 C++ FastLaunchPlan。stock 2.10.0.post6 构建所用 PTA 为 38e483ee，隔离包将上游绑定注册到 torch_npu._C。两腿均加载同一路径且具有真实 plan 和 submit 符号。[源码与构建](../pr42634/overlay/README.md) · [上游固定提交](https://gitcode.com/Ascend/pytorch/commit/2701f7b3e8d89625752edbde10e42c2b2172272c)

原始 01 动态 eager 本轮正序两腿各 60 步，入口计数均为零，torch_npu._inductor.config 和 patch 模块均未加载。开关从 0 变为 1 时同步窗口实际从 565.304 变为 605.396 ms/步。这个区间反映未激活配置的波动。Level0 的 A/B、B/A 已分别归档，反序同步窗口未成功完成。原始 Level1 的第 4、5 步各有 9,892、9,893 次 ACLNN enqueue，匹配的 Inductor wrapper 调用名均为零。[A 腿计数](working/evidence/pr42634-original-no-profiler-ab/01-A/rank000/pr42634_receipt.json) · [B 腿计数](working/evidence/pr42634-original-no-profiler-ab/02-B/rank000/pr42634_receipt.json) · [两序 Level0 与已完成的同步窗口](reports/pr42634-original/measurement.md) · [原始 Level1](../pr42634/eligibility/stock-level1.json)

另行适配的 eager fold 对 E01 每步一次的有序累加调用显式提供 ABI 和资源信息。8 步训练门 B 腿建立 1 个计划，首调用走普通 Triton，3 步因物理 grid 上限回退，4 步真实进入并返回 C++ 提交；A/B 输入、梯度、权重、优化器状态及非有限跳步通过。该项独立于原版 wrapper 自动触发路径。训练门给出等价性和命中次数，正式比较另取无 profiler 及 Level0。[适配源码](pr42634_eager_fold) · [动态逐步命中](working/gates/pr42634-eager-v3/gate-attempt-5taqpl7q/02-B/rank000/eager_fold_plan.json) · [语义对照](working/gates/pr42634-eager-v3/gate-attempt-5taqpl7q/comparison.json)

## 尝试无优化路径

- **因果 mask 改为 BF16 tril。** 12 个微测样例的输出和梯度逐位一致，四个微测窗口节省 19.17%–31.18%。完整训练正序 21.208 → 22.588 ms/步，慢 6.51%，该轮 Free 占比 88.43% → 86.51%。[微测](working/diagnostics/tril-micro/micro.json) · [整步八列](reports/causal-tril-v1/measurement.md)

- **融合 causal SiLU attention。** 完整融合的 14 个 NPU 逐位样例均出现差异。保留原矩阵乘和 SiLU、只融合掩码和除 L 的实现通过 14/14 例，但四个前反向同步窗口均慢 22.83%–40.01%。[完整融合原始差异](working/diagnostics/attention-full/micro.json) · [局部融合实测](working/diagnostics/attention-partial/micro.json)

- **RMSNorm 融合。** 原生 NPU RMSNorm 与整段 Triton RMSNorm 均未通过 BF16 位级门；保留原归约、只融合相邻乘法的方案通过正确性，但四个微测窗口约从 558–624 增至 902–919 μs。[局部融合](working/diagnostics/diag-rms-v1/micro.json) · [整段融合](working/diagnostics/rms-full/micro.json)

- **分行 flat stats。** 算子微测通过，8 步完整训练门在第 7 步更新后状态发生差异。该路径停止性能归档。[微测](working/diagnostics/row-micro-v1/micro.json) · [训练门差异](working/gates/row-stats-v1/gate-attempt-ire85eb2/comparison.json)

- **重复使用 CANN executor 的 embedding 提交包。** 三个实际形状输出逐位一致。[1,19]、[2,14]、[2,22] 的局部同步窗口中位数依次为 26.404 → 27.190、26.876 → 26.263、25.505 → 24.561 ms，未得到一致改善。[各腿原始数据和请求数](working/diagnostics/emit-micro/micro.json)

- **侧表反向分组。** 按 16、64、256、1,024 张表分组，在四种形状下检查梯度一致。Level0 三个形状对应的 1,024 表整组均快于 256 表分组；[4,31] 上两者为 1.348、1.308 ms。[分组微测](working/diagnostics/tiling-micro/micro.json)

## 仍需训练窗口验证的 Host 开销

flat_stats.eligible 对每个 1,052 片梯度重新核对视图根、真实 storage、dtype、shape、连续性和 offset。实机独立微测的 Python/C++ Host 中位为 2,031.527/56.593 μs，p95 为 2,055.543/69.222 μs，5 次 native guard 未产生 ATen 事件。缩短 Host 检查能否提前后续有效提交仍取决于同一步、同流的空档和正式 A/B。[实机微测](working/diagnostics/native-guard-micro/micro.json) · [实现与守卫](native_guard/README.md)

在 packed 梯度上进一步改为直接计算 norm，已完成的正序 Level0 中 Free 占比 91.32% → 92.76%；缺少同配置同步计时，不能判断整步收益。[Level0 分项](reports/flat-norm-v1/measurement.md)

## 代码、原始回执与复算

每项代码和实机索引按本轮独立分支归档，早期动态 D01–D05 位于上一层台账。当前快照含 3,738 个原始条目的 SHA-256，涵盖 workspace、runner、设备前后检查、同步窗口、官方 CSV、归因摘要和数值门。[代码分支](https://github.com/j1204150587-svg/pytorch/tree/codex/npu8-dynamic-eager-pr42634-20261127) · [实现与原始证据提交](https://github.com/j1204150587-svg/pytorch/commit/ecae80312b4c5cc84615807637bf0c870237ea6a) · [收据清单](working/receipt-sha256.json) · [完整 trace 路径和 SHA-256](working/trace-index.json)

在 dynamic-free70 目录复算一项对照。代码从对应 session 的实际命令、计数、shape、CSV 和指标读取数值。

~~~sh
python3 -B runner_tools/report.py runner_tools/recipes/side-slab-v1.json --receipts working/evidence --schedule working/evidence/grouped-side-v1-no-profiler-ab/workspace/configs/dynamic_shapes_npu8.json --output reports/side-slab-v1
~~~
