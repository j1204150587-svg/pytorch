# 前向 embedding 与 mean 在 consumer 上的就绪时间

原双向 embedding 配方的 Level1 第 4、5 步为 [1,19]、[2,14]。每步按原主线程提交次序检出 1,024 对 `aclnnEmbedding` 和 `aclnnMean`。每对均对应物理流 47 上的 AsStrided、Gather 和 Mean，合计每步 3,072 笔。复算使用同一条 Host 时间线的 enqueue、dequeue 及对应设备任务。[逐笔 CSV](requests.csv.gz) · [复算脚本](availability.py)

| consumer 开始处理 embedding 后的时间 | [1,19] 已完成 mean enqueue | [2,14] 已完成 mean enqueue |
|---|---:|---:|
| 0 µs | 0 / 1,024 | 6 / 1,024 |
| 10 µs | 0 / 1,024 | 11 / 1,024 |
| 20 µs | 902 / 1,024 | 939 / 1,024 |

以 enqueue 返回作为已发布的保守判据，mean 的中位发布时间分别晚 15.300、12.905 µs。若只要求 mean 已进入 enqueue，consumer 入口两步仍分别为 0 和 6 笔，等待 10 µs 时为 0 和 216 笔。可见这两个 trace 在 consumer 入口允许立即前瞻的数量极少。主动等待会延迟首笔 embedding 提交，其端到端代价尚无 A/B。发现相邻算子也不能证明中间张量无其他读者，替换两项工作还需要独立的 eager 语义依据。本项只做时序归因。[形状、各时间门限和 trace 指纹](results.json)
