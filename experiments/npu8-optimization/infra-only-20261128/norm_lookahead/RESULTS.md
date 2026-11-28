# 实测 Cast→LpNorm 队列前瞻上限

保留原训练、ZeRO、动态 eager 及 runner；沿用 01 overflow、双向 embedding 描述符和 65,536 项缓存的真实 B 腿 Level1。两步的形状是 [1,19]、[2,14]。本次没有新增上卡采样，结果仅覆盖此两步。原始 trace 记录于 `20261128T043752Z-ab-l1-7m6ais34/02-B`。[原始 trace](../evidence/raw/20261128T043752Z-ab-l1-7m6ais34/02-B/trace_view.json.gz) · [事件映射与原始分位表](../embedding/analysis/emb-fwd/B)

## 候选机制与证伪条件

DeepSpeed `runtime/zero/stage_1_and_2.py::get_grad_norm_direct` 按梯度执行 `torch.linalg.vector_norm(g.data.double().detach(), ord=norm_type)`。该段在一个生产线程 334627、一个消费线程 334652 和物理流 47 上，每步产生 1,052 对相邻 Cast 与 LpNorm。Cast 的 PTA 队列记录是 `aclnnInplaceCopy`，设备任务是 `aclnnInplaceCopy_CastAiCore_Cast`；Norm 对应 `aclnnLinalgVectorNorm` 和 `aclnnLinalgVectorNorm_LpNormV2_LpNormV2`。各记录恰有一次 `aclrtLaunchKernelWithHostArgs`，各对的两个物理槽相邻。每步还有一个段外 LpNorm，未纳入。

候选是在第一个 Cast 出队时，消费线程立即查看后续已发布的工作项；若 Cast→Norm 的类型、流、参数和使用者均兼容，可考虑一次调用消除一次提交。跨两对的候选还要求第二对已经发布。证伪条件是满足无等待前瞻的频率过低，或没有证据保证写入、异常及同流可见性语义。任一步需等生产线程发布下一项，即会推迟原 Cast 的提交。

## 发布边界与实际设备提交

生产者 enqueue 开始是最乐观的界，入队结束是相对保守的界；PTA trace 没有独立标出槽位实际发布的指令。分母为同对 1,052、下一对 1,051。表格每格为该消费者观察时刻已开始 enqueue / 已结束 enqueue 的同对 Norm 数量。物理开始和 ACL 返回两行用于显示过晚决策会放大可见数量。

| Cast 消费者或设备时刻 | 第 4 步 | 第 5 步 |
|---|---:|---:|
| Cast dequeue 开始 | 1 / 1 | 2 / 2 |
| Cast ACLNN 调用开始 | 1 / 1 | 2 / 2 |
| Cast 实际 ACL launch API 开始 | 4 / 1 | 2 / 2 |
| Cast 实际 ACL launch API 返回 | 163 / 45 | 6 / 5 |
| Cast 硬件开始执行 | 1,003 / 887 | 1,047 / 1,045 |

从 Cast 出队开始直至 ACL launch API 入口，两个步骤已发布的下一对首个 Cast、下一对的 Norm 均为 0/1,051；enqueue 开始和结束两界结果一致。Cast 出队时，本 consumer 上已开始入队且尚未结束出队的请求数，在两步的 p50/p95/p99 均是 1、最大 2，其中包括当前 Cast。改用入队结束为发布界，分位数相同，第 4 步最小值为 0，说明个别当前请求的 enqueue 与 dequeue 标记发生交叠。整段 2,104 个出队记录连续，无其他请求的 enqueue 开始/完成标记，无 trace 可见的 ACL 同步、事件记录、等待或查询。设备物理顺序、同流及未看到额外标记，都无法证明临时 Cast 输出独占，或所有可能别名、写入及外部流访问均不可观察。

| 第一步 Cast 出队后，等到请求发布的时间（µs，p50 / p95 / p99 / max） | 第 4 步 | 第 5 步 |
|---|---:|---:|
| 本对 Norm，enqueue 开始 | 12.54 / 25.20 / 29.15 / 182.17 | 14.82 / 23.28 / 28.87 / 211.99 |
| 本对 Norm，enqueue 完成 | 15.08 / 30.58 / 34.81 / 189.82 | 19.04 / 27.33 / 34.63 / 217.95 |
| 下一对 Cast，enqueue 完成 | 45.17 / 64.22 / 74.04 / 231.34 | 44.53 / 59.94 / 81.27 / 254.58 |
| 下一对 Norm，enqueue 完成 | 66.75 / 99.23 / 108.34 / 258.13 | 66.16 / 89.70 / 123.12 / 275.85 |

等待按 max(0, 发布时刻−观察时刻) 逐行计算，分位取升序下标 floor((n−1)·p)。[逐对数据](pairs.csv.gz) · [全部发布边界、在途计数和分位数](summary.json)

## 与设备空档的交集

按 Cast dequeue 开始时 Norm 已完成 enqueue 的保守条件，第 4 步只命中对序号 239、流 47 槽 14467–14468；第 5 步只命中 157 和 161、槽 14304–14305 与 14312–14313。对应两项任务所引入的设备空档合计为 35.78、44.40 µs，其中与原 Cast 或 Norm 自身 ACL launch 执行区间相交的仅为 0、3.03 µs。相同两步全部 1,052 对的 Cast 前空档分别为 27.152、23.973 ms，Norm 前空档为 26.306、25.612 ms。

假设每个符合条件的对消除原 Cast launch，旧 Cast API 服务时间合计为 5.66、7.61 µs；若把每对两项中较贵的一次视为可省，仍只有 6.42、8.59 µs。此数衡量已有单项 API 服务，没有计入新融合核的服务时间，也不能直接换算整步收益。已开始但尚未完成 enqueue 的槽位不能直接读取。即便将两端都放宽到 Cast 的 ACL 入口、Norm 的 enqueue 开始，理论上也仅发现 4、2 对；这几对原 ACL 时间与空档之交为 16.45、3.03 µs。

两步共 2,104 对，保守无等待窗口仅 3 对有机会合并，最多消除 3 次原物理 launch。通用 `copy_` 路径允许可观察的写入，要进一步证明 Cast 输出独占或保留其可观察结果，还需满足原 dtype 转换、归约精度、异常时序。当前不实施融合或跨对批处理，归入「尝试无优化路径」。没有生成端到端或 Free 的新 A/B 结果。

```text
python -B norm_lookahead/lookahead.py embedding/analysis/emb-fwd/B evidence/raw/20261128T043752Z-ab-l1-7m6ais34/02-B/trace_view.json.gz norm_lookahead
```
