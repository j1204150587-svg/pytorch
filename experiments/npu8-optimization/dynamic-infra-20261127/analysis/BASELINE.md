# 动态 eager 的提交与设备空档

逻辑 NPU 8，1,024 张侧表，单 rank、梯度累积 1、BF16 ZeRO-2。两步均启用 01 分块 overflow 和 Task Queue Mode 1。Level1 #4 和 #5 实际输入分别为 `[1,19]` 和 `[2,14]`。按训练进程内已执行的形状，两步均为此 `[B,L]` 首次出现。下表为本轮实机基线。[实测形状](../evidence/baseline-l1/rank000/metrics.json) · [启动命令](../evidence/baseline-l1/command.json) · [原始 trace的宿主路径与哈希](../evidence/raw-files.md)

| 主动步 | 设备计算核 / SDMA | ACL kernel / 同步 memcpy / 异步 memcpy API | 链到后继任务的内部工作空档 | 未链接内部工作空档 | 官方 Stage / Free（ms） |
|---|---:|---:|---:|---:|---:|
| #4 | 15,093 / 2,127 | 15,093 / 6 / 2,121 | 17,213 段，639.111 ms | 6 段，0.864 ms | 729.615 / 674.234 |
| #5 | 15,094 / 2,127 | 15,094 / 6 / 2,121 | 17,214 段，736.601 ms | 6 段，1.177 ms | 791.474 / 739.683 |

内部工作空档指全部物理流上两段相邻计算、AICPU 或搬运之间的空闲，只采两端同落在当步 Host marker 内的区间。Level1 的官方 Free 还涉及不同的边界与任务分类。每段用唯一的 Hardware task、HostToDevice 两端、CANN connection、嵌套 ACL API 和 PTA 队列 correlation ID 匹配。部分 ACL 调用与设备执行重叠，按设备任务开始截断。ACL 返回到设备开始的时段混合依赖、设备排队和时钟对齐误差。全部原始流顺序指纹和未关联时长均保留。[硬件流、提交数与长尾](baseline-l1/submissions.json) · [官方 CSV](../evidence/baseline-l1/step_trace_time.csv) · [逐笔联结](baseline-l1/device_gaps.csv.gz)

## 重复出现的 producer 热点

每步 1,027 个 `aten::embedding_dense_backward` 请求触发 6,159 次 kernel launch。1,026 笔各可见一次 producer 内的 `aclnnEmbeddingDenseBackwardGetWorkspaceSize`。下表为同线程嵌套时长，单位 µs。队列等待取 `dequeue.start − enqueue.end` 的带符号值。consumer 包含内部 API。producer 从同线程最内层 ATen 入口计至 enqueue 开始。[逐算子分位数](baseline-l1/operator_latency.csv) · [逐笔队列事件](baseline-l1/queue_calls.csv.gz)

| 时间边界 | #4 p50 / p95 / p99 / max | #5 p50 / p95 / p99 / max |
|---|---:|---:|
| ATen 入口至入队开始 | 71.91 / 130.19 / 146.91 / 390.41 | 95.54 / 180.50 / 213.71 / 447.79 |
| 其中 GetWorkspaceSize | 50.35 / 94.45 / 109.47 / 365.04 | 66.69 / 129.93 / 157.33 / 296.30 |
| 可见队列等待 | 2.34 / 2.90 / 3.15 / 6.96 | 6.17 / 7.15 / 7.77 / 24.02 |
| consumer dequeue 服务 | 124.08 / 143.45 / 150.69 / 178.20 | 159.61 / 179.98 / 197.75 / 296.03 |
| 内部单次 kernel launch API | 4.98 / 7.61 / 9.37 / 24.92 | 5.23 / 8.10 / 9.99 / 28.27 |

后继为 embedding 反向的工作空档合计 183.928 / 246.259 ms。按每个后继的时间线互斥切分，与其 producer 的 workspace API 重合约 51.7 / 57.5 ms，与其可见队列等待重合约 2.5 / 6.2 ms，与其单次 ACL launch API 重合约 6.3 / 5.6 ms。前向 `aclnnEmbedding` 同样每步有 1,027 次请求，1,026 次可见 workspace API，workspace 与后继设备工作空档重合约 21.4 / 8.4 ms。该重合量限定可继续定位的窗口，移除 workspace 后能回收多少 Free 需要配对测量。[按算子与线程的空档分解](baseline-l1/gap_attribution.csv)

**原生反向候选。** `StructKernelNpuOpApi.cpp::embedding_dense_backward` 对 indices 建立局部描述符范围。op-plugin 的 `add_param_to_buf(const at::Tensor&)` 把张量的 shape、stride、storage_offset 和 storage 长度放入 executor key。`ConvertType(const at::Tensor&)` 创建对应的 ACL 描述符。业务侧 `feature_ids[:, :, table_index, :]` 的每张侧表相差 4 个索引元素。同一步内存在可共用 shape、stride 和地址对齐类别的只读视图。候选为已核验的基格式 int64 三维视图，以 512 字节为单位前移 ACL base，同时更新 residual offset 及覆盖的 storage 长度，对 hash 和 ACL 描述符作同一变换。训练张量继续持有原 storage。用每步命中/回退、GetWorkspaceSize 数、梯度和真实提交序列验证。若 workspace 未减少、出现错误或端到端不能重现收益，分别记录。[源码生成脚本](../rebase/apply_rebase.py)

## 队列、D2D 与步尾

每步各 9 次 `aclrtSynchronize*`，全部 API 墙钟合计 0.126 / 0.131 ms，最长一次 32.94 / 36.88 µs。这给出改变这些 API 自身时长的上限。排空前的 PTA 等待以及未打点的 queue release 需另查。主线程的 `aclnnInplaceCopy` 所关联的工作空档为 101.741 / 116.720 ms，混合 D2D 和 dtype 转换；其中 56.6 / 87.7 ms 出现在对应 ACL API 返回之后，剩余才有可拆的 producer/consumer 等阶段。按物理流和设备任务类别拆分可从逐笔文件重算。[空档分解](baseline-l1/gap_attribution.csv) · [同步和步尾](baseline-l1/drain_and_tail.csv)

将全部已链接空档与后继请求的等待阶段相交，#4/#5 共 13.294 / 19.926 ms。其中前序已标记的 consumer dequeue 涵盖 0.356 / 0.819 ms，位于已记服务之外的有 12.938 / 19.107 ms。写指针发布、线程可运行时间、wait/wakeup 和 release 的先后还需局部打点。[逐算子等待交集](baseline-l1/queue_wait_overlap.csv)

新动态基线每步有 3,218 个 `aclnnInplaceCopy` dequeue，其中 2,112 个各严格关联一笔 `aclrtMemcpyAsync` 和真实 SDMA，内部 kernel 为零。下表将这些 D2D 单列，各步另有 8 次前反向 clone/masked_fill 的复制。已匹配的是源码守卫筛选前的数量。分配流和地址条件要在实际变体内查验。[逐笔验证](baseline-l1/d2d_joins.csv) · [分组分位数、互斥空档交集](baseline-l1/d2d_subgroups.csv) · [调用记录](baseline-l1/d2d_calls.csv.gz)

| 按 #4 / #5 展示 | 反向 worker 的 AccumulateGrad | 主线程反向尾部连续 D2D |
|---|---:|---:|
| 每步笔数 | 1,052 / 1,052 | 1,052 / 1,052 |
| copy 入口至入队 p50（µs） | 25.95 / 33.10 | 15.15 / 21.00 |
| 队列等待 p50（µs） | 22.98 / 47.11 | 2.08 / 6.35 |
| consumer 服务 p50（µs） | 10.68 / 15.63 | 9.55 / 20.43 |
| 单次 memcpy API p50（µs） | 5.70 / 6.09 | 4.53 / 6.02 |
| 对应设备工作空档（ms） | 31.78 / 44.00 | 66.61 / 76.48 |
| 其中 workspace API（ms） | 1.82 / 1.07 | 6.07 / 1.66 |
| 其中队列等待（ms） | 0.55 / 0.73 | 2.34 / 3.70 |
| 其中 dequeue 开始至 memcpy 开始（ms） | 0.95 / 1.12 | 3.35 / 8.09 |
| 其中 memcpy API 返回后（ms） | 22.10 / 36.43 | 29.54 / 50.69 |

主线程的长连续 SDMA 段有动态重测依据。#44 增量需同轮固定 #42 并取得实际跨分配流守卫计数。worker 的 workspace 与空档的交集只占该组 5.7% / 2.4%，而排队等待绝对分位数随 shape 的变化大于其真实空档暴露。历史固定形状 #42 两轮无 profiler 均缩短，Level0 Free 方向相反，A/A 相差 43.68 ms/步。[固定形状 #42 实测](../provenance/prior-materials.md)

两步最长内部空档均出现在 worker 下一次 `aclnnMul` 入队前，分别 14.27 / 14.73 ms。#4 的 autograd `AccumulateGrad` 覆盖其中 14.14 ms，可见 1,052 次 `view_as`。次长内部空档在后继 `aclnnCat`，长 10.62 / 11.25 ms；#4 的 `flatten_dense_tensors` 覆盖 10.05 ms，内部有 1,052 个 view 和 format cast。随后单次 Mul 或 Cat 的 ACL API 对应时间需单列。#4 末设备任务到 #5 首任务空档 32.295 ms，其中 17.515 ms 位于权重恢复的 `aten::unflatten_dense_tensors`，全在官方 Free；#5 权重恢复 17.840 ms，期间未见设备工作，官方 Free 轨在 profiler 最后活跃步尾已截断。[长尾内的 CPU scope](baseline-l1/largest_gap_cpu_anchors.csv) · [末核和 unflatten](baseline-l1/drain_and_tail.csv)

**反向 hook 查询参与训练的参数。** 将上述最长空档中第一个非负 autograd 序列号的 `aten::view_as` 起点至最后一个终点作为观测窗。#4 / #5 各有 1,052 个连续序列号（4208–5259 / 5260–6311），每笔内嵌一次 `aten::view`。观测窗 13.689 / 14.081 ms 全落在官方 Free，所有硬件流上的工作交集为零，这一 worker 在窗内无 queue 或 ACL 提交。1,052 个 `view_as` 墙钟和为 5.597 / 5.317 ms，其余 8.092 / 8.764 ms 位于相邻调用之间。父线程在下次 Mul 提交前有这段开销，无法把它归到 Mul 的 ACL launch 或队列消费。[逐步复算 CSV](baseline-l1/leaf_view_tail.csv)

DeepSpeed 0.19.0 的 `grad_handling_hook` 在更新 hook 计数时调用 `count_used_parameters_in_backward`，该函数逐个查叶子参数的 autograd 节点，再由 `_will_engine_execute_node` 核验当前 backward graph 中的参与者。PyTorch 2.10.0 的 `_get_grad_fn_or_grad_acc` 对需梯度的叶子调用 `t.view_as(t).grad_fn.next_functions[0][0]`。源码与上述序列号、频次和无提交特征一致。Level1 未采 Python 栈，不能由这份 trace 证明调用者或把 inter-call 时间全部分给 Python。若要减少节点查询次数，须保留动态控制流、unused parameters、reentrant backward 和 hook 重入语义，并超出本轮 TorchNPU、op-plugin 和 CANN 提交层范围。[ZeRO hook](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.0/deepspeed/runtime/zero/stage_1_and_2.py#L1062-L1069) · [逐参数检查](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.0/deepspeed/runtime/utils.py#L1449-L1474) · [PyTorch 叶子节点查询](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/autograd/graph.py#L183-L193)

## 复算与历史边界

在本目录运行。分位数取排序后的下标 floor(q × (n−1))。每行 CPU 时间窗可相互嵌套。单条物理流内按任务起点、名称和 Task Id 稳定排序；物理流重编号时可用完整序列 SHA-256 的多重集比较。

```text
python -B historical_level0.py ../../dynamic-shapes-20261127/evidence historical_level0.csv
python -B analyze_dynamic_trace.py ../evidence/baseline-l1 baseline-l1
python -B inspect_tail.py ../evidence/baseline-l1 baseline-l1
python -B leaf_view_tail.py ../evidence/baseline-l1 baseline-l1
python -B queue_wait.py ../evidence/baseline-l1 baseline-l1
python -B d2d_subgroups.py ../evidence/baseline-l1 baseline-l1
```

历史三次动态 A 腿的 Level0 #4–6 各有相同的三个 shape。九个步中反向墙钟 434.47–524.76 ms，反向内全部流没有计算、AICPU 和搬运的时段累计 398.45–491.77 ms。反向 **主线程** 匹配的 1,072 个 enqueue 中，相邻 consumer 间隙落在下一次 enqueue 开始前的部分累计 46.34–64.54 ms。该统计遗漏 autograd worker 的 producer；与本轮全线程 Level1 分开解读。[历史逐步、分阶段 CSV](historical_level0.csv)
