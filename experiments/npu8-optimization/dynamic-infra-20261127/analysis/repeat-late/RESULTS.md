# 同一实际形状首次执行和再次执行

默认动态 eager + 01、Task Queue Mode 1、默认 executor 缓存容量。标准序列的 Level1 #4、#5 首次遇到 `[1,19]`、`[2,14]`。另一个训练进程按前八种形状连续跑两轮，第 12、13 步再次遇到这两种形状并采 Level1。两次训练的 RNG 和参数状态随之前各步演进，本对照用来量化两种缓存状态下的算子边界。另有同一进程内 8+8 的实际查找回执，八种形状首次执行时前向、反向各 8,216 笔中仅命中 5 笔，第二轮各 8,216 笔全部命中。[标准序列原始 trace的宿主路径与哈希](../../evidence/raw-files.md) · [8+8 原始 trace的宿主路径与哈希](../../evidence/raw-files.md) · [同进程实际命中](../../evidence/repeat/per-producer/01-A/repeat-op-totals.csv)

## 同一算子、流和提交数

两个 shape 的各 producer 线程 ATen 顺序、ACL 提交计数及流 44、45、47 的设备工作序列哈希均匹配。[逐形状校验](first-repeat-work.json)

| 实际 shape | 主动步 | 全步 kernel launch | 同步 / 异步 memcpy | 前向 embedding 请求 / launch | 反向 embedding 请求 / launch |
|---|---|---:|---:|---:|---:|
| [1,19] | #4、#12 | 各 15,093 | 各 6 / 2,121 | 各 1,027 / 2,051 | 各 1,027 / 6,159 |
| [2,14] | #5、#13 | 各 15,094 | 各 6 / 2,121 | 各 1,027 / 2,051 | 各 1,027 / 6,159 |

第 12 步额外出现一个 `PLACE_HOLDER_SQE` 控制项。第 13 步所有已记录硬件类型也与第 5 步相同。[首次序列](../baseline-l1/submissions.json) · [再次执行的序列](submissions.json)

## producer 准备、consumer 服务与 launch

下表为微秒 p95，每格按“首见 → 再见”排列。每步的前向及反向 embedding 各有 1,026 笔可见 `GetWorkspaceSize`，再次执行时均为 0 笔。前向/反向可见 workspace API 的总占用在首次执行 [1,19] 时为 25.266 / 59.094 ms，[2,14] 时为 25.664 / 78.752 ms。[各阶段的次数和 p50/p95/p99](first-repeat-latency.csv)

| 算子和 shape | ATen 入口至入队 | enqueue→dequeue 等待 | dequeue 服务 | 单笔实际 ACL kernel launch |
|---|---:|---:|---:|---:|
| 前向，[1,19] | 82.20 → 25.25 | 2.81 → 3.40 | 60.66 → 23.43 | 6.39 → 6.94 |
| 前向，[2,14] | 81.66 → 22.61 | 2.86 → 8,433.61 | 60.73 → 57.26 | 6.67 → 11.36 |
| 反向，[1,19] | 130.19 → 21.65 | 2.90 → 13.52 | 143.45 → 75.89 | 7.61 → 8.55 |
| 反向，[2,14] | 180.50 → 24.39 | 7.15 → 7.37 | 179.98 → 62.84 | 8.10 → 7.73 |

第 13 步前向排队 p99 为 8.935 ms。将排队与该步骤后继为前向 embedding 的 2,051 个设备空档逐笔对齐，相交总计仅 1.496 ms；该组后继空档共 35.729 ms。较长的等待包含队列中已有工作和其他请求的运行。首见与再见时，后继为前向/反向 embedding 的空档分别从 **99.208 / 183.928** 降至 **50.530 / 55.912 ms**（[1,19]），从 **100.336 / 246.259** 降至 **35.729 / 60.404 ms**（[2,14]）。[可见等待的重叠](queue_wait_overlap.csv) · [初次空档联结](../baseline-l1/gap_attribution.csv) · [再次空档联结](gap_attribution.csv)

## 形状复用后仍可观测的空档

下列时间为同一步上跨所有流的工作空档，按结束空档的下一次提交分组。后继工作名称与导致空档的上游函数需逐笔区分。

| 后继请求与 producer | #12 [1,19] | #13 [2,14] | 可定位的环节 |
|---|---:|---:|---|
| `InplaceCopy`，主线程 | 120.678 ms | 129.946 ms | 与本次 workspace API 相交 6.16 / 4.73 ms，consumer 到 API 起点 15.89 / 12.50 ms。API 返回至设备开始仍有 50.26 / 74.88 ms，含流依赖、设备排队和对时误差 |
| `Divs`，worker | 101.940 ms | 113.203 ms | 在该笔 ATen 入口之前已有 39.74 / 38.25 ms，入口至 enqueue 余段 14.65 / 16.19 ms，后续等待与空档交叠 4.72 / 5.08 ms |
| `InplaceCopy`，worker | 65.251 ms | 74.129 ms | 与 workspace 相交 7.33 / 4.66 ms，API 返回至设备开始 20.35 / 34.93 ms |

两步最长单段空档均结束于 worker 的 `aclnnMul`，分别为 16.346 / 19.357 ms；在对应 ATen 入口前已过去 16.270 / 19.237 ms。第 12 步的 `AccumulateGrad` 与该空档相交 16.224 ms，其中可见 1,052 次 `view_as`。该步另有一段后继为 `Cat` 的 7.906 ms 空档，优化器 `flatten_dense_tensors` 与其相交 7.177 ms。#41 实机探针已说明梯度切片在调用顺序上不相邻。[具体 step、队列 ID 和 flow](submissions.json) · [CPU scope 交集](largest_gap_cpu_anchors.csv) · [#41 守卫证据](../../flatten/RESULTS.md)

Level1 官方 Stage / Free 在第 12 步为 610.770 / 554.275 ms，第 13 步为 661.514 / 608.111 ms。上述内部空档的边界和任务分类与官方 Free 有差异。[官方 CSV](../../evidence/repeat-late/01-A/step_trace_time.csv)

### 第 13 步前向排队长尾

从对应线程的 enqueue 结束量到 dequeue 开始，合并墙钟上彼此重叠的等待区间。在再次遇到 `[2,14]` 的第 13 步，1,027 笔前向 embedding 中有 820 笔等待超过 1 ms。逐请求累加为 5,054.781 ms，墙钟并集为 79.508 ms。下面比较这批请求与同一形状首见时的情况。上文的队列、ACL 和设备工作数量核验在两个步骤均成立。[逐组复算](queue_backlog.csv)

| 同一形状前向 embedding | 首见 #5，所有正等待 | 再见 #13，所有正等待 | 再见 #13，等待超过 1 ms |
|---|---:|---:|---:|
| 笔数 / 当步 1,027 笔 | 1,023 | 1,027 | 820 |
| 等待的墙钟并集（ms） | 2.630 | 84.988 | 79.508 |
| 其中 consumer 已在其他已标记 dequeue 内（ms） | 0.191 | 62.754 | 59.574 |
| 其中全部设备流都没有计算或搬运（ms） | 2.552 | 73.399 | 68.630 |
| 已与同一请求的下一个设备工作关联的等待空档（ms） | 2.292 | 1.496 | 0.865 |

第 13 步长等待并集内，consumer 已在执行前序 embedding、Mean 和 Add 的标记范围各占 25.748、18.419 和 15.407 ms。其余 19.934 ms 分布于这些标记之间，最长连续区间为 89.85 µs。前序 dequeue 的 59.574 ms 中，有 51.638 ms 与设备空闲重合，说明应观察更早的前序提交。将当步所有正等待都计入，能直接联到本请求下一个设备工作的等待空档为 1.496 ms。临近请求共用在途队列，逐请求相加会重复同一段墙钟。仅凭现有 dequeue 活动标记还不能判定线程在这些范围内是否被抢占或阻塞。[前序服务按算子分解](queue_backlog_services.csv) · [从两份原始 trace 复算](../revisit_queue_budget.py)

复算从 `analysis` 目录运行。

```sh
python -B analyze_dynamic_trace.py ../evidence/repeat-late/01-A repeat-late
python -B queue_wait.py ../evidence/repeat-late/01-A repeat-late
python -B inspect_tail.py ../evidence/repeat-late/01-A repeat-late
python -B ../cache/compare_shape_visits.py baseline-l1 repeat-late
python -B revisit_queue_budget.py ..
```
