# D05 前向 embedding 描述符规范化的 Level1 对照

默认动态 eager + 01、队列模式 1，两腿打开同一构建并单切 `NPU_DYNAMIC_REBASE_FORWARD`。第 4、5 步分别首次出现 `[1,19]`、`[2,14]`。每步 1,027 笔前向 embedding 均按同线程上的 ATen 序号、队列 correlation 和设备 flow 匹配。每笔请求的 API 序列、实际物理流 ID 和工作任务的流内位置均相同。每步 1,024 个侧表请求中，第一个设备核由 Slice 改为 AsStrided。producer 准备和 consumer 服务缩短，设备核实现也发生变化。[A 命令](../../evidence/forward-rebase-l1-ab/01-A/command.json) · [B 命令](../../evidence/forward-rebase-l1-ab/02-B/command.json) · [逐笔核验](embedding/counts.csv)

| 指标（µs） | #4 A → B 的 p50 | #4 A → B 的 p95 | #5 A → B 的 p50 | #5 A → B 的 p95 |
|---|---:|---:|---:|---:|
| ATen 入口至 enqueue 开始 | 62.98 → 16.95 | 95.64 → 29.37 | 64.60 → 18.65 | 89.99 → 39.13 |
| enqueue 标记范围 | 8.81 → 9.05 | 10.48 → 11.46 | 8.96 → 9.66 | 11.47 → 14.49 |
| enqueue 结束至 dequeue 开始 | 6.18 → 8.19 | 23.53 → 10.30 | 6.98 → 8.22 | 341.91 → 9.66 |
| dequeue 服务 | 78.52 → 23.36 | 99.47 → 33.91 | 79.90 → 24.36 | 110.63 → 31.21 |

每步可见的 `aclnnEmbeddingGetWorkspaceSize` 从 1,026 减到 18 次。第 4 步这些调用共计 37.435 → 0.484 ms，第 5 步为 39.299 → 0.481 ms。队列等待的第 5 步 p99 从 1,029.78 降到 27.23 µs，最大值从 1,136.35 降到 55.08 µs。对 2,051 笔真实 `aclrtLaunchKernelWithHostArgs` 逐笔统计，两步 p95 分别从 9.35 升到 10.68 µs、从 7.57 升到 9.68 µs。这次可证实的提速发生在 ACL launch 外的 Host 路径。[逐笔请求与延迟](embedding/latency.csv) · [单笔 ACL launch 分位数](operator_latency.csv)

每个侧表的两个设备 kernel 都保留。A 腿每步 1,024 个侧表请求均可见两次 CANN tiling。B 腿其中 1,008 个无可见 tiling，其余 16 个仍各有两次。第 4 步 A 侧表的 ACLNN 内部 p50 为 70.36 µs，B 无可见 tiling / 仍有 tiling 的两组为 19.29 / 65.36 µs。第 5 步依次为 71.11、20.10 / 66.21 µs。这些切片也随请求次序、cache 状态和设备实现变化。[A 的 ACLNN 嵌套计时](A/embedding_consumer.csv) · [B 的 ACLNN 嵌套计时](B/embedding_consumer.csv)

| 空档与 Host 阶段的交集（ms） | #4 A → B | #5 A → B |
|---|---:|---:|
| 结束于 embedding 后继设备核的全部空档 | 104.894 → 55.977 | 97.757 → 65.533 |
| 其中与 producer 的 workspace API 相交 | 8.849 → 0.117 | 2.785 → 0.040 |
| 与该请求可见队列等待相交 | 5.276 → 7.736 | 2.093 → 6.485 |
| 与 dequeue 开始至该次 ACL API 相交 | 28.438 → 5.735 | 18.775 → 5.695 |
| 与该次 ACL API 自身相交 | 11.957 → 8.829 | 6.557 → 8.350 |

空档从所有物理流上前一项计算或搬运工作结束，量至后继 embedding 核开始，每步 2,051 段。以对应后继请求为准，空档开始时已在执行 producer workspace API 的数量，第 4、5 步分别为 345 → 7 和 47 → 4；已 dequeue、尚未到该次 ACL API 的为 784 → 30 和 507 → 78。第 5 步 A 的排队大长尾只与这些后继空档交叠 2.093 ms，无法把全体队列等待时长等同于可消除的设备空闲。[全部互斥阶段](gap_attribution.csv) · [A 的空档起点](A/first_visible.csv) · [B 的空档起点](B/first_visible.csv)

全步第 4 / 5 步 kernel launch 为 15,093 / 15,094，均另有 2,121 次异步 memcpy 和 6 次同步 memcpy。流 44 的 1,053 个、流 45 的 8 个工作任务全部按槽位和名称一致。流 47 的第 4 / 5 步工作任务分别为 16,159 / 16,160 个，每步只发生上述 1,024 个同类型核名替换。ATen 序列及 ACL 提交计数相同，PTA 的 `destroy_event` 队列项相差 −3 / +1。[全步核对](submissions_compare.json) · [双向逐流核对](physical_stream_substitutions.json)

独立 Level1 采样中官方 Free 第 4 步为 710.230 → 634.139 ms，第 5 步为 714.112 → 657.221 ms。八列性能表使用独立的无 profiler、Level0 成对结果。[A 原始 CSV](../../evidence/forward-rebase-l1-ab/01-A/step_trace_time.csv) · [B 原始 CSV](../../evidence/forward-rebase-l1-ab/02-B/step_trace_time.csv)

从 `analysis` 目录运行以下命令可复现上述数据。

```sh
pair=forward-rebase-l1-ab
python -B compare_dynamic_legs.py ../evidence/$pair/01-A ../evidence/$pair/02-B $pair
python -B compare_operator_calls.py $pair --aten aten::embedding --op aclnnEmbedding --out $pair/embedding
python -B stream_substitutions.py $pair
python -B aclnn_service.py ../evidence/$pair/01-A $pair/A/embedding_consumer.csv --op aclnnEmbedding
python -B aclnn_service.py ../evidence/$pair/02-B $pair/B/embedding_consumer.csv --op aclnnEmbedding
python -B first_visible.py $pair/A
python -B first_visible.py $pair/B
```
