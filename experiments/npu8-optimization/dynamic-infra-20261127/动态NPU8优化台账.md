# 动态形状 HSTU 训练的 NPU 8 Infra 优化台账

## 负载与独立基线

2026-11-27，宿主逻辑 NPU 8 映射到板 4、芯片 0、物理卡 8，训练容器 `recsys-gr-ds019-npu8` 内为 `npu:0`。每腿使用宿主共享锁，进程检查按芯片区分。BF16 ZeRO-2，单 rank，梯度累积 1，1,024 张侧表，seed 1234，Task Queue Mode 1，开启 01 分块 overflow。除列明的原生前置开关外均使用原分区和 norm，HSTU 前反向全程 eager。

原工作区可用的是 60 个互异的合成 `[batch_size, seq_len]`，batch size 1–4、序列长度 12–32。每步直接生成该尺寸的输入。相同步号的 A/B 共享形状和随机输入顺序。无 profiler 窗口首尾同步，第 0–9 步预热，第 10–59 步共 136 个样本、2,994 个 token。吞吐以窗口工作量除以总秒数计算。[实际 shape 序列](evidence/baseline-steady/workspace/configs/dynamic_shapes_npu8.json) · [启动命令与卡况](evidence/baseline-steady/01-A/command.json)

本轮独立 stock A 为 **600.060 ms/步，4.533 样本/s，99.790 token/s**。Level0 第 4–6 步分别为 `[1,19]`、`[2,14]`、`[2,22]`，Free **652.088**、Stage **703.152**、Computing **50.921 ms/步**，ΣFree/ΣStage 为 **92.738%**。所有优化行使用各自同轮 A 腿。最后一个活跃步的 CPU ProfilerStep 与官方 Stage 存在截断差，按官方 CSV 完整保留第 6 步。[同步计时](evidence/baseline-steady/01-A/rank000/metrics.json) · [逐步 Level0](evidence/baseline-l0/01-A/step_trace_time.csv)

旧台账的图回放、图内侧表、回放 CPU 亲和性、回放排空等路径依赖图模式。旧 #30–36 涉及每流队列、Mode 2 轮询、绑核、日志取时、autograd 线程和直接回调，固定形状试验已记录持平、回退或波动。已有动态训练复测确认 07 梯度打包和分块 norm 可以改变反向执行量。本轮选择原 ZeRO 分区和 norm，使两腿保留逐参数复制与同样的查找负载。D04 两腿共同开启 #42。[固定形状台账](provenance/prior-materials.md) · [原始训练调用链](provenance/prior-materials.md) · [已有动态训练三组对照](provenance/prior-materials.md)

各表首行是正序 A/B，次行是反序 B/A，数值都按 A→B 展示。端到端降幅为 `1−B步时/A步时`，最后一列为 `1−B的Free占比/A的Free占比`，均从未舍入数值计算。每项只报告相对其同轮 A 腿的增量；组合收益需要独立组合配对。

## 动态调用链

每个 Level1 活跃步有 1,027 个 embedding dense backward 请求、6,159 次内嵌 kernel launch。动态基线第 4、5 步中，设备所有流的工作空档与反向 GetWorkspaceSize 的交集分别为 51.74、57.46 ms。这里统计的是时间交集，各阶段存在重叠。9 次同步 API 各步合计约 0.13 ms，不能解释上述空档。队列等待含在途工作及线程调度，可见的等待长尾和实际设备空档分别核对。[基线的分段方法与分位数](analysis/BASELINE.md) · [各候选触发和证伪条件](候选与证伪.md)

在默认容量下，把前 8 个实际形状在同一进程内再走一遍。前、反向 embedding 的查找都从首次 8,216 笔中的 5 次命中，变为再见时 8,216 次全命中。独立 Level1 按同一形状核对，`[1,19]` 和 `[2,14]` 的两类 embedding 各从 1,026 次可见 workspace 准备降至 0 次，设备计算与搬运数相等。再见 `[2,14]` 时，820 笔超过 1 ms 的前向排队等待合计 5,054.781 ms，墙钟并集 79.508 ms，其中 59.574 ms 已在前序 consumer 服务中；直接联到这些请求后继设备空档的等待只有 0.865 ms。复访采样的训练状态、步号与正式首次采样不同。[同形状的算子、流和长尾](analysis/repeat-late/RESULTS.md)


## 有实测收益的优化

### D01 扩大 CANN executor 缓存

进程启动时将 `ACLNN_CACHE_LIMIT` 从 stock 默认改为 `65536`，作用于 Mode 1 的 `PTAGetExecCache`。每个新训练 shape 内，`Divs`、`Mean` 和 `Add` 仍有大量相同的算子键。50 步内总查询次数均为 494,640，命中由 115,305 增至 280,960。新增命中有 99.84% 发生在同一步内。诊断进程末步 RSS 增加约 530 MiB。[实测分算子缓存行为](cache/RESULTS.md) · [配置与实现](cache-limit.json)

| 优化前端到端时间 | 优化后端到端时间 | 端到端提升比例 | 优化前 Free 时间 | 优化前 Free 占比 | 优化后 Free 时间 | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 726.677 ms/步 | 555.665 ms/步 | +23.53% | 631.262 ms/步 | 92.41% | 621.247 ms/步 | 92.32% | +0.10% |
| 774.679 ms/步 | 529.083 ms/步 | +31.70% | 694.372 ms/步 | 93.07% | 585.279 ms/步 | 91.84% | +1.32% |

两行依次为 A/B 和 B/A，每步分别节省 171.012 和 245.596 ms。正序吞吐从 3.743 增至 4.895 样本/s，82.403 增至 107.763 token/s。反序从 3.511 增至 5.141 样本/s，77.297 增至 113.177 token/s。端到端降幅相差 8.17 个百分点。Level0 第 4–6 步总共只多 2 次缓存命中，早窗 Free 差异缺乏容量机制的直接解释。[逐步 Stage、Computing、Free 与端到端复算](reporting/cache-limit/RESULTS.md) · [8 步语义与注入检查](evidence/cache-gate-v2/comparison.json)

在缓存容量已产生差异的 Level1 #12、#13，Divs、Mean 和 Add 的逐笔物理流、任务槽位、核名与数量相同。每步有 1,039 个 worker Divs 请求，producer p95 从 40.47/40.54 降至 17.96/19.73 µs，consumer p95 从 43.68/46.26 降至 22.40/22.83 µs。该组后继设备工作空档从 132.68/134.34 降至 53.42/40.47 ms。其他算子的空档随提交时机变化，后段全步 Level1 Free 只降低 22.403/1.307 ms。[后段逐流归因](analysis/cache-limit-l1-late-ab/RESULTS.md)

### D02 规范化 embedding 反向索引的 ACLNN 描述符

`op_api::embedding_dense_backward` 为合格的只读索引切片建立临时描述符，`add_param_to_buf` 与 `ConvertType` 使用同一组基址、余数 offset 和可达长度。每次按 512 字节对齐移动描述符基址，原存储仍拥有数据。8 步训练中有 8,192 个侧表请求命中、24 个回退，合格请求仅构造 128 个描述符。独立微测覆盖多形状、非零 offset、末表、重复 ID、padding 及回退；真实训练覆盖三个非有限跳步和五次权重更新。[实现](rebase/apply_rebase.py) · [实测代码提交](https://github.com/j1204150587-svg/pytorch/commit/22ad9642075624016afe445f280feca24610665e)

正序与反序分别少 111.576 和 53.088 ms/步。Level0 Free 正序增加 45.483 ms/步，反序减少 32.952 ms/步，标准早窗的 Free 改善未复现。[逐腿 Stage、Computing、吞吐及波动](reporting/index-rebase/RESULTS.md)

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 606.854 ms/步 | 495.278 ms/步 | +18.39% | 589.394 ms/步 | 91.87% | 634.877 ms/步 | 92.55% | −0.73% |
| 530.369 ms/步 | 477.281 ms/步 | +10.01% | 693.014 ms/步 | 92.95% | 660.062 ms/步 | 92.70% | +0.27% |

第 4、5 步各有 1,027 个反向 embedding 请求、6,159 次 kernel launch。每笔物理流和流内位置一致。每步 1,024 个内部核从 Slice 换为 AsStrided；其他工作核名保持一致。producer 准备 p50 从 76.04/74.45 降至 15.36/15.21 µs，consumer 服务 p50 从 158.18/150.15 降至 59.28/68.77 µs；ACL launch 自身 p95 升高。同一算子的 workspace 与全流空档交集从 48.343/29.199 ms 降至 0.422/0.036 ms。[逐笔与逐流核对](analysis/index-rebase-l1-ab/RESULTS.md)

### D05 规范化前向 embedding 的权重与索引描述符

在 op-plugin 的 `op_api::embedding` 中对合格的 BF16 权重和索引视图同时规范化描述符及 executor 键。两腿使用同一独立库，开关 `NPU_DYNAMIC_REBASE_FORWARD` 从 0 切至 1，反向 D02 保持关闭。8 个实际形状共命中 8,192 个侧表请求、回退 24 个，16,384 次规范化键操作只构造了 288 个描述符。逐步 1,052 个梯度、AdamW 状态、权重和三次溢出跳步核对通过。[前向实现](rebase_forward/apply_forward.py) · [微测](evidence/forward-micro/02-B/micro.json) · [真实训练](evidence/forward-gate/comparison.json)

正序少 88.424 ms/步，反序少 98.654 ms/步，两次端到端降幅相差 0.84 个百分点。两序的 Free 占比都略降，绝对 Free 时间正序少 56.050 ms/步、反序多 14.523 ms/步。[整步、吞吐与各步 Free / Stage / Computing](reporting/forward-rebase/measurement.md)

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 681.861 ms/步 | 593.436 ms/步 | +12.97% | 789.730 ms/步 | 93.74% | 733.680 ms/步 | 93.32% | +0.45% |
| 714.650 ms/步 | 615.996 ms/步 | +13.80% | 745.266 ms/步 | 93.54% | 759.789 ms/步 | 93.36% | +0.20% |

第 4、5 步前向每步仍有 1,027 次请求和 2,051 次 kernel launch，流和流内位置相同。每步 1,024 个核由 Slice 换成 AsStrided，workspace API 从 1,026 次降至 18 次，producer p95 由 95.64/89.99 降到 29.37/39.13 µs，consumer p95 由 99.47/110.63 降到 33.91/31.21 µs。单个 ACL launch p95 在两步都升高。[逐笔核对与空档](analysis/forward-rebase-l1-ab/RESULTS.md)

D02 和 D05 的 8,192 次都是输入通过快路径守卫的计数。128 和 288 次分别是合格张量进入 `ConvertType` 的次数，D05 的权重、索引按两个张量计；实际 workspace API 次数另由 Level1 测得。单卡实测覆盖 Mode 1、非 capture 的 NPU int64 `[B,L,4]` 只读索引，D05 的权重为连续 BF16 `[64,64]`。60 个形状跑完整训练性能窗，其中前 8 个形状逐步比对全部梯度、权重和溢出跳步，独立微测另含 rank-2 回退。异设备、其他队列模式和 capture 的调用没有在本轮上卡验证；D02 和 D05 各用独立构建。

## D03 修订的实机验证

TorchNPU 包安装时为 `torch._utils._unflatten_dense_tensors` 增加窄适配入口。B 遇到连续 NPU BF16 flat 和合格的 BF16 Meta 模板时调用独立 C++ op；每个模板按正确 offset 只建立一个新的 `as_strided` 视图。A 和 B 均关闭旧 #43 和 #41。8 步中共有 5 次正常更新、3 次非有限跳步，B 的 13 次调用对应 5 次真实权重回写激活、8 次单模板桶回退。每步 1,052 个梯度以及 BF16/FP32 权重、AdamW 状态均核对通过。独立微测覆核非零 offset、版本计数、存储别名与后注册显式 AutogradMeta 的回退。[TorchNPU 入口](meta_unflatten/adapter.py) · [构建收据](provenance/meta/build-receipt.json) · [正确性](reporting/meta-views/RESULTS.md)

第 4、5 步每步减少 3,156 次 Host ATen 操作，保留 1,052 次 `as_strided`，三条物理流的设备提交次数、名称与顺序均相同。权重恢复大组分别从 19.712 → 4.506、25.191 → 5.526 ms；两腿这段时间内全设备均无计算和搬运。第 4 步末任务至下一步首任务的空档从 33.782 降至 20.413 ms。第 5 步跨步空档超出采集边界。[逐步视图、物理流和设备空档](analysis/meta-views-l1-ab/RESULTS.md)

首轮无 profiler 正序 744.559 → 726.741 ms/步（+2.39%），对应 3.653 → 3.743 样本/s、80.423 → 82.395 token/s。13:05 逻辑卡 8 被外部进程占用，反序在第一腿启动前退出，追加的 A/A 在第一腿运行后卡况核对未通过。两组无效记录已排除，等待独占后重跑。本项目前确认局部 Host 和设备空档改善，完整端到端及 Level0 结论待配对。[同步窗口与实测状态](reporting/meta-views/RESULTS.md)

14:11（UTC+8）复核时，板 4 芯片 0 仍有两个外部进程，SGLang 调度进程占显存 45,107 MiB。训练容器仅有原来的 `sleep`，共享锁已释放。恢复控制器在宿主等待，缺少本配方的无 profiler 反序、Level0 正反序以及独立无 profiler A/A、A/B、B/A。以本轮每个完整配对 137–157 秒计，合计约需 15 分钟独占时间。[卡况快照](evidence/occupied-slot.json) · [两次无效运行的原始回执索引](provenance/interrupted-pairs.json)

恢复后每完成一对，宿主都从成功的运行目录重算并写入 `/data/recsys-gr-ds019-host88/artifacts/dynamic-infra-20261127/reporting/meta-views/measurement.md`、`eight_columns.md` 与 `paired_results.json`。已用完成的第一对和 Level1 验证该入口，当前只产生一对稳态结果，待同序 Level0 补齐后才生成相应八列。[宿主复算器](reporting/host_windows.py)

## 尝试无优化路径

### D04 跨分配流直接 SDMA 提交，整步收益未复现

两腿均开启既有的同流 #42。op-plugin 在 `copy_d2d_baseformat_opapi` 中只为符合条件的跨流源切换路径。B 先由 PTA allocator 的 `recordOtherStreamCopySource` 记录源块的使用流，再由 `copy_d2d_by_memcpy` 提交。独立微测验证跨流同步、源生命周期与回退，8 个 shape 的梯度、overflow 跳步、权重和 AdamW 均通过。压力测试中移除调用方的 stream 记录并立即释放源时，A 可过早复用源块；真实训练配对没有观察到由此造成的错误。[补丁](cross_copy/from-stock.patch) · [实机微测](evidence/cross44-micro/02-B/micro.json) · [真实训练语义](evidence/cross-gate/comparison.json)

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 671.667 ms/步 | 668.477 ms/步 | +0.47% | 638.591 ms/步 | 92.43% | 757.029 ms/步 | 93.62% | −1.29% |
| 644.590 ms/步 | 599.690 ms/步 | +6.97% | 781.432 ms/步 | 93.88% | 786.968 ms/步 | 93.87% | +0.01% |

额外两对无 profiler 复测中，正序 694.429 → 613.964 ms/步（+11.59%），反序 631.395 → 684.630 ms/步（−8.43%）。同期 A/A 为 649.705 → 684.452 ms/步，同一实现相差 34.747 ms/步。正式 Level0 两序的绝对 Free 各增加 118.438、5.537 ms/步。[两轮同步计时、A/A、吞吐与逐步 Free](reporting/cross-copy/measurement.md)

#4/#5 的主线程各有 1,052 个跨分配流请求；逐笔对齐后，每笔在同一物理流 44 的相同槽位保留一次 memcpy 和一次 SDMA。workspace 从每步 1,052 次降至 0。producer p50 从 25.27/25.90 降至 5.49/4.57 µs，consumer p50 从 17.10/19.53 降至 6.44/6.59 µs，目标 SDMA 前的全设备工作空档合计各少 15.630/13.309 ms。单个 memcpy API p95 两步都升高。局部准备和服务的提速成立，整步和 Free 未通过复测。[同流同槽位逐笔关联](analysis/cross-copy-l1-ab/RESULTS.md)

### D03 原版 #43 未在动态训练中激活

A/B 语义门加载的 `libtorch_npu.so` 均为 `36faab10…`。B 的开关为 1，第 4、5 步 `NPUUnflattenContiguousViews` 均为 0。两腿大组都执行 narrow、slice、as_strided、view 各 1,052 次。步骤 #4 的父 scope 从 24.013 变为 9.132 ms，#5 则从 21.278 变为 24.083 ms；这两笔差异不能记到 #43。发现未激活后停止后续配对，独立诊断真实 flat、模板和 dispatcher。[原始失效配对](analysis/unflatten-l1-ab/RESULTS.md) · [实测库](evidence/unflatten-gate/comparison.json)

真实权重回写以 NPU BF16 flat 和 1,052 个 Meta 模板调用 unflatten。8 步里有 8 次大组和 8 次单模板桶，常规派发均未经过原来只注册在 AutogradPrivateUse1 的实现；显式派发该键的阳性对照可命中。旧同设备守卫又排除了 Meta，据此单独立项检验 D03 修订。[实机派发诊断](unflatten_probe/RESULTS.md)

在停止旧版控制器前已经启动的一对无 profiler 训练完成，630.535 → 628.926 ms/步，因未激活而不归为此改动的收益。[已完成窗口](evidence/unflatten-inactive-ab/status.json)

### 原版 #41 整段 flatten

8 个实际形状共执行 16 次 flatten，快路径 0 命中。8 次梯度桶各只有一张量，另外 8 次的 1,052 个分片共享 storage，但按调用列表顺序有 1,051 个偏移不满足相邻条件，每次仍发一个 Cat。跳过该 Cat 做整段 clone 会改变梯度排列。[逐次布局和计数](flatten/RESULTS.md)

## 剩余瓶颈

- `AccumulateGrad` 之后、下一个 `aclnnMul` 入队之前，#4/#5 的 1,052 个连续 `view_as` 从首个到末个历时 13.689/14.081 ms，全部在官方 Free 内，期间各硬件流没有工作，worker 也没有发出队列项或 ACL。可见 `view_as` 自身累计 5.597/5.317 ms，调用之间累计 8.092/8.764 ms。DeepSpeed 的 used-parameter 计数和 PyTorch 的 leaf grad-accumulator 查询与这个模式相符，当前两步 Level1 缺少 Python 调用栈。[逐步复算及对应源码](analysis/BASELINE.md) · [原始区间统计](analysis/baseline-l1/leaf_view_tail.csv)
- 默认 A 的 `aclrtSynchronize*` 每步 9 次，调用自身耗时合计 0.126/0.131 ms。#4/#5 中可见队列等待与后继全流空档只相交 13.294/19.926 ms。[队列与排空分解](analysis/BASELINE.md)
- 独立 stock Level1 在私有 tracefs 实例中仅跟踪实际 consumer TID，109,993 条目标事件零丢失。#4/#5 的后继工作空档中，位于前序 dequeue 之外的排队交集为 32.952/19.068 ms；其中线程已阻塞的时间界于 2.279–23.097/1.108–15.656 ms，已唤醒而等待 CPU 的时间界于 0.367–13.434/0.391–8.822 ms。对时交集宽 5.340 µs，tracefs 分辨率 1 µs，各状态上下界分别求得。与 ACL API 重合的空档中，至少 65.523/65.812、29.784/29.803 ms 的 consumer 已调度到 CPU。内核可运行状态已可区分，PTA 写指针发布、eventfd 等待、服务后释放及 CANN 内部时段仍缺逐调用专用打点；这批证据尚未锁定一个可单独验证的队列改动。[线程调度与原始区间](analysis/scheduler-l1/RESULTS.md)
- 形状再见时，#12/#13 的主线程 `InplaceCopy`、worker `Divs` 和 worker `InplaceCopy` 是三组最大的后继空档，分别为 120.68/129.95、101.94/113.20 和 65.25/74.13 ms。单次 ACL 返回到实际任务开始之间还混有流内依赖、设备排队和跨时钟误差，无法直接算作可回收的 Host 时间。[再见 shape 余下的热点](analysis/repeat-late/RESULTS.md)

## 实测代码与复算

所有配方在独立分支 `codex/npu8-dynamic-infra-20261127`。下表链接固定到包含实测实现的提交 `841ee04475b462a3df7de65e282ee96fa74079d5`。PTA 源码基于 `38e483ee79ec8a3286517808740a811c0f0ae66c`，op-plugin 基于 `eaf6b55812a1998a4a40e485e3b2861a40e9c2f6`。库 SHA-256 用于核对 A、B 加载了同一构建。[实验分支](https://github.com/j1204150587-svg/pytorch/tree/codex/npu8-dynamic-infra-20261127) · [实现提交](https://github.com/j1204150587-svg/pytorch/commit/841ee04475b462a3df7de65e282ee96fa74079d5)

| 尝试 | 启动开关或改动入口 | 构建或运行指纹 |
|---|---|---|
| D01 CANN 容量 | [cache-limit.json](https://github.com/j1204150587-svg/pytorch/blob/841ee04475b462a3df7de65e282ee96fa74079d5/experiments/npu8-optimization/dynamic-infra-20261127/cache-limit.json) | [stock 库](provenance/stock-library.json) |
| D02 后向索引描述符 | [apply_rebase.py](https://github.com/j1204150587-svg/pytorch/blob/841ee04475b462a3df7de65e282ee96fa74079d5/experiments/npu8-optimization/dynamic-infra-20261127/rebase/apply_rebase.py) | [构建收据](provenance/rebase/build-receipt.json) |
| D03 旧版 unflatten | [TorchNPU 原始补丁](https://github.com/j1204150587-svg/pytorch/blob/841ee04475b462a3df7de65e282ee96fa74079d5/experiments/npu8-optimization/dynamic-infra-20261127/provenance/unflatten/unflatten.patch) | [构建收据](provenance/unflatten/build-receipt.json) |
| D03 修订 | [包内 Python 入口](https://github.com/j1204150587-svg/pytorch/blob/841ee04475b462a3df7de65e282ee96fa74079d5/experiments/npu8-optimization/dynamic-infra-20261127/meta_unflatten/adapter.py) · [受守卫的 C++ 视图](https://github.com/j1204150587-svg/pytorch/blob/841ee04475b462a3df7de65e282ee96fa74079d5/experiments/npu8-optimization/dynamic-infra-20261127/meta_unflatten/meta_unflatten.cpp) | [sidecar 与 adapter 指纹](provenance/meta/build-receipt.json) |
| D04 跨分配流复制 | [PTA 与 op-plugin 补丁](https://github.com/j1204150587-svg/pytorch/blob/841ee04475b462a3df7de65e282ee96fa74079d5/experiments/npu8-optimization/dynamic-infra-20261127/cross_copy/from-stock.patch) | [构建收据](provenance/cross/build-receipt.json) |
| D05 前向描述符 | [apply_forward.py](https://github.com/j1204150587-svg/pytorch/blob/841ee04475b462a3df7de65e282ee96fa74079d5/experiments/npu8-optimization/dynamic-infra-20261127/rebase_forward/apply_forward.py) | [构建收据](provenance/forward/build-receipt.json) |
| #41 旧版 flatten | [布局观测](https://github.com/j1204150587-svg/pytorch/blob/841ee04475b462a3df7de65e282ee96fa74079d5/experiments/npu8-optimization/dynamic-infra-20261127/flatten/inspect_flat.py) | [原始布局](evidence/flatten-probe.jsonl) |

从本目录运行 `python reporting/recompute_recipe.py <配方名>` 可按命令、形状表、计时窗口和官方 CSV 重新生成表格。每个分析报告附逐笔归因的复算命令。完整 trace 和原始 profiler 表保存在本地 `evidence/` 与宿主 `/data/recsys-gr-ds019-host88/artifacts/dynamic-infra-20261127/`，可以按文件大小和 SHA-256 校验。[计时与正确性复算](reporting/recompute_pairs.py) · [宿主路径与哈希索引](evidence/source-index.json) · [复现说明](README.md)

PTA 调度采样器、对时上下界算法、追加 A/A 与重复正反序配置另有固定提交。[采样与复核代码 3563d43](https://github.com/j1204150587-svg/pytorch/commit/3563d43a0fef0b836df19b8bfc83b96c9be54bfc) · [完整 trace 与查找记录清单](evidence/raw-files.md)
