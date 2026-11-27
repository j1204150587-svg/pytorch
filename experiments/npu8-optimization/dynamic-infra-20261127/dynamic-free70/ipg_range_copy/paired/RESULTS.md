# IPG 梯度桶在两个动态形状上的实际提交与空档

正序 Level1 对比中，第 4 步 `[1,19]` 和第 5 步 `[2,14]` 均首次遇到该实际形状。每步完成一个含 1,052 个 BF16 梯度、4,380,544 个元素的桶。A 逐参数调用 `aten::copy_`，B 将已归约的连续范围作为一次 `aten::copy_` 提交给 PTA。B 的八步均命中，回退为零。语义门逐步核对 1,052 个梯度、权重、优化器与跳步，#2、#4、#6 跳过更新，其余五步更新；最大 loss 绝对误差 0.000003815，norm 误差 0。

流 44 的第 0 个设备工作在两腿均为 `Divs`。A 的第 1–1,052 个工作各对应一个 `Enqueue@aclnnInplaceCopy`、一个 `aclrtMemcpyAsync` 和一个 SDMA。B 的第 1 个工作对应一次 `Enqueue@aclnnInplaceCopy`、一次 `aclrtLaunchKernelWithHostArgs` 和一个 `aclnnInplaceCopy_TensorMoveAiCore_TensorMove`。其 enqueue 完整落在主线程 `NPUIPGBucketRangeCopy` 内。流 45、47 在两腿的所有控制和工作任务序列各自相同。每步少 1,051 个设备工作，另多一次 enqueue record_event、对应流 44 的 EVENT_RECORD，所有 PTA 队列项净减少 1,050。每步 ATen 的 view、view_as 和 copy_ 分别少 2,104、1,052 和 1,051 次。[A 逐步计数](01-A/copy-census.json) · [B 逐步计数](02-B/copy-census.json)

同一步两腿的全设备工作时间线用相同的两个端点对齐。从物理流 45 的第 1 个工作 `MEMCPY_ASYNC` 结束，至物理流 47 的 `ZerosLike` 开始（#4 为槽位 13,833，#5 为 13,834，槽位自 0 起）。此间所有流上只有目标复制工作。先按设备工作并集计算空档，再逐笔关联 PTA/CANN。B 的两个 TensorMove 设备时长分别为 12.18、13.12 µs，Host marker 分别长 72.95、80.81 µs。[物理端点、覆盖、逐段归因](range-intervals.json)

| 步/腿 | 复制前各段全流空档之和 ms | 复制设备时间 ms | 末次复制至 ZerosLike 空档 ms | 对齐区间全流空档 ms | 对齐区间墙钟 ms |
|---|---:|---:|---:|---:|---:|
| #4 A | 56.887 | 0.675 | 7.091 | 63.978 | 64.653 |
| #4 B | 18.737 | 0.012 | 10.879 | 29.616 | 29.628 |
| #5 A | 56.390 | 0.674 | 7.063 | 63.452 | 64.126 |
| #5 B | 18.793 | 0.013 | 11.100 | 29.893 | 29.907 |

对齐区间的全流空档分别少 34.362、33.559 ms。B 剩余复制前空档分别有 18.683、18.702 ms 发生在 `aten::copy_` 入口之前。全步 Level1 官方 Free 分别从 564.324 降至 534.899 ms、从 552.493 增至 561.639 ms；局部节省与整步变动可同时观察，正式 Free 使用独立三步 Level0。

B 的复制前空档中，每步主线程出现 1,053 次 `narrow` 和 1,052 次 `view`，对应一段源视图与 1,052 段目标视图的构造。所有可见 ATen 范围按并集分别覆盖 7.516、7.648 ms。复制后的空档中，两步各有 1,052 次 `_has_compatible_shallow_copy_type`，与逐梯度发布 `g.data` 的实现对应。该段所有可见 ATen 范围并集只覆盖 0.567、0.662 ms，其余 10.312、10.438 ms 还包含 Python 发布、记录和后继优化器准备。逐函数分摊需要额外打点。[CPU 区间与计数](range-intervals.json)

| 步/腿 | ATen 至 enqueue µs | enqueue µs | 队列等待 µs | consumer 服务 µs | ACL API µs |
|---|---:|---:|---:|---:|---:|
| #4 A，每列为 1,052 笔的 p95 | 16.31 | 6.68 | 2.69 | 11.20 | 4.91 |
| #4 B，单笔 | 12.55 | 8.87 | 1.91 | 14.93 | 7.51 |
| #5 A，每列为 1,052 笔的 p95 | 15.37 | 7.04 | 2.62 | 10.68 | 4.93 |
| #5 B，单笔 | 13.11 | 9.88 | 6.83 | 20.75 | 9.09 |

A 的单笔 ACL p99 为 7.81/8.10 µs，最长 9.23/10.52 µs；B 选择了不同的设备复制实现。收益机制是训练框架在 PTA 调用之前合并已知的连续源和连续目标，减少 Host 调用和设备工作次数。首次/再见的独立 stock 观察均保留 1,052 笔连续复制。[独立首次与再见对照](../README.md)

在动态实验根目录先对原始两腿逐一执行 `python dynamic-infra-20261127/analysis/analyze_dynamic_trace.py <原始腿目录> <分析腿目录>`，再执行 `python dynamic-free70/ipg_range_copy/submission_census.py <分析腿目录> <分析腿目录>/copy-census.json`。端点和空档用 `python dynamic-free70/ipg_range_copy/range_interval.py <含01-A及02-B的原始配对目录> <分析配对目录> <输出json>` 复算。程序校验前后端点的流、槽位、工作名称、步号、形状、区间内任务顺序，以及每个区间的工作和空档之和。
