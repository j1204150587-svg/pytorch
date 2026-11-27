# D02 反向 embedding 描述符规范化的 Level1 对照

两腿使用相同动态 shape、Mode 1、01 overflow 和同一构建。#4 为 `[1,19]`，#5 为 `[2,14]`。规范化后每步仍有 1,027 个 `aten::embedding_dense_backward`、6,159 个该算子的 ACL kernel launch。每步各 1,024 个侧表请求把设备内部的 `Slice` 换成 `AsStrided`。本轮证实 producer 参数准备和 consumer 服务缩短，单个 ACL kernel launch 的 p95 在两步均上升。[正式 A 腿](../../evidence/index-rebase-l1-ab/01-A/command.json) · [正式 B 腿](../../evidence/index-rebase-l1-ab/02-B/command.json)

## 算子、物理流与内部实现

每步全线程的 ATen 次数、每个线程的 ATen 顺序以及 ACL 各类提交次数在两腿相同。#4 / #5 的全步 kernel launch 分别为 15,093 / 15,094，两个步骤均有 2,121 次异步 memcpy 和 6 次同步 memcpy。PTA 控制队列项 `destroy_event` 的 B−A 差值分别为 −5 和 +6。[完整逐步提交核对](submissions_compare.json)

对每个物理流的工作任务按时间排序并分配从 0 开始的流内序号。在 `(step, Physic Stream Id, slot)` 上分别由 A 找 B、由 B 找 A。两个方向均无漏项。再由每个 worker 的 ATen 调用序号，经 enqueue/dequeue correlation、唯一的 CANN HostToDevice flow、真实硬件任务，验证两腿每一笔 embedding 反向的流 ID 和流内位置。每步 1,027 个请求均匹配，同一位置所有内部核名也相同的有 3 笔，其余 1,024 笔各改变一个核名。[双向逐槽位结果](physical_stream_substitutions.json) · [逐请求计数](embedding/counts.csv) · [逐请求延迟和流位置核对](embedding/matched_calls.csv.gz)

| 物理流 | #4 工作任务，A=B | #5 工作任务，A=B | 两步逐槽位差异 |
|---|---:|---:|---|
| 47 | 16,159 | 16,160 | 两步均有 1,024 个 `AI_VECTOR_CORE` 的 `aclnnEmbeddingDenseBackward_SliceAiCore_Slice` 换为 `aclnnEmbeddingDenseBackward_AsStridedAiCore_AsStrided`，其余位置相同 |
| 44 | 1,053 | 1,053 | 核名、类型和顺序全部一致 |
| 45 | 8 | 8 | 核名、类型和顺序全部一致 |

以全流序列的完整名称 SHA 匹配时，流 47 的指纹随核名改变。因此逐笔结果中严格序列指纹相同的请求数为零；按实际物理流 ID 和流内位置核验的结果保存在表格相邻两列。将每个流上的实际核名进一步拆开，可以判定本轮同时存在 executor 缓存命中、内部 tiling 减少和设备实现改变。

## 逐笔耗时与对应的空档

下表单位为 µs，每格为 A→B。两步各含 1,027 个请求，内部 launch 按 6,159 笔计算。分位数按排序下标 floor(q × (n−1))。队列等待为 enqueue 完成至 dequeue 开始的带符号差。dequeue 内部包含 CANN 调用，其时长不能再与 launch 相加。[按算子的全量分位数和长尾](operator_latency.csv) · [按逐笔匹配的 A/B 统计](embedding/latency.csv)

| 边界 | #4 p50 | #4 p95 | #5 p50 | #5 p95 |
|---|---:|---:|---:|---:|
| ATen 入口至 enqueue 开始 | 76.04→15.36 | 109.83→18.95 | 74.45→15.21 | 109.95→73.80 |
| 可见队列等待 | 4.39→7.91 | 5.39→31.49 | 4.43→29.14 | 5.73→17,482.15 |
| consumer dequeue 服务 | 158.18→59.28 | 208.82→99.97 | 150.15→68.77 | 199.79→180.59 |
| 单次 ACL kernel launch | 5.80→5.67 | 9.78→11.24 | 5.55→6.14 | 9.40→14.13 |

工作区查询次数 #4 为 **1,026→18**，#5 为 **1,026→54**。在每笔发六个核的 1,024 个侧表调用中，A 均有五次可见 tiling；B 中无 tiling 的有 1,008 / 972 笔，仍有五次 tiling 的有 16 / 52 笔。保留的两笔其余 GetWorkspaceSize 来自非侧表的五核请求。仅对剩余查询取时，不能估算被消除的查询量。[A 腿 consumer 分解](A/embedding_backward_consumer.csv) · [B 腿 consumer 分解](B/embedding_backward_consumer.csv)

B 腿内部可在相同 shape、相同六个物理核名称和顺序下按当笔是否进入 workspace API 分组。#4 有查询的 16 笔 / 无查询的 1,008 笔，producer p50 为 86.48 / 15.35 µs，consumer p50 为 183.46 / 59.05 µs。#5 的 52 笔 / 972 笔分别为 88.85 / 15.15 µs 和 276.64 / 67.07 µs。这是单腿内的观测，组间还存在执行次序与排队深度差异。[同核序列内按 workspace 分组](B/side_table_cohorts.csv)

第 4、5 步每腿均有 6,159 个内部设备工作空档可精确链接到该算子的后继 kernel。按所有流计算、AICPU 或搬运的并集，空档总和分别从 **213.312→57.926 ms** 和 **206.695→75.823 ms**。其中与 producer workspace API 的交集是 **48.343→0.422 ms**、**29.199→0.036 ms**。与后继请求的队列等待交集是 **4.568→1.012 ms**、**4.732→0.618 ms**。#5 B 的等待长尾延伸到了该笔后继任务的直接前置空档之外，期间还可能有其他请求的设备空档。相同空档与 ACL launch API 的交集为 12.100→6.453 ms 和 6.609→4.431 ms，交集由供给时机共同决定。[互斥空档分解](gap_attribution.csv)

在后继为 embedding 反向的 6,159 个空档开始时，#4 A 有 **942** 次首个可见阶段是该笔 producer 正在执行 workspace API，B 降为 **11** 次；#5 为 **964→1** 次。#4 A/B 分别有 3,833 / 4,861 个空档起点晚于该任务的 ACL API 返回，#5 为 5,083 / 5,669 个。后一类仍需设备侧排队、执行依赖及跨时钟观测。[A 腿空档起点](A/first_visible.csv) · [B 腿空档起点](B/first_visible.csv)

两步 Level1 官方 Free 分别为 **635.434→567.943 ms** 和 **621.436→571.333 ms**。这些数用于本次调用归因。端到端和标准化 Free 表应分别引用无 profiler 和 Level0 正反序。[A 腿原始 CSV](../../evidence/index-rebase-l1-ab/01-A/step_trace_time.csv) · [B 腿原始 CSV](../../evidence/index-rebase-l1-ab/02-B/step_trace_time.csv)

在上级 analysis 目录复算。

```text
python -B compare_dynamic_legs.py ../evidence/index-rebase-l1-ab/01-A ../evidence/index-rebase-l1-ab/02-B index-rebase-l1-ab
python -B compare_operator_calls.py index-rebase-l1-ab --aten aten::embedding_dense_backward --out index-rebase-l1-ab/embedding
python -B stream_substitutions.py index-rebase-l1-ab
python -B aclnn_service.py ../evidence/index-rebase-l1-ab/01-A index-rebase-l1-ab/A/embedding_backward_consumer.csv
python -B aclnn_service.py ../evidence/index-rebase-l1-ab/02-B index-rebase-l1-ab/B/embedding_backward_consumer.csv
python -B workspace_cohorts.py index-rebase-l1-ab/B aclnnEmbeddingDenseBackward index-rebase-l1-ab/B/side_table_cohorts.csv
python -B first_visible.py index-rebase-l1-ab/A
python -B first_visible.py index-rebase-l1-ab/B
```
