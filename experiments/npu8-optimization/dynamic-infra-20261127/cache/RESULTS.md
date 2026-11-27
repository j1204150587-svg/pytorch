# 动态 HSTU 的 CANN executor 缓存命中

同一 60 步实际形状序列、1,024 张侧表、动态 eager、01 分块 overflow、Task Queue Mode 1。A 沿用默认环境，B 设 `ACLNN_CACHE_LIMIT=65536`。两腿均用同一动态链接审计库，记录 `CanUsePTACache` 的可用性及 `PTAGetExecCache` 实际返回的 executor。下表对齐零基步号。该诊断的运行时间不参与稳态测速。[A 启动命令](../evidence/cache-diag/01-A/command.json) · [B 启动命令](../evidence/cache-diag/02-B/command.json) · [源码机制和复核方法](README.md)

| 步窗口 | A/B 各自查找次数 | A 命中 | B 命中 | A 命中率 | B 命中率 | B 每步多命中 |
|---|---:|---:|---:|---:|---:|---:|
| 第 4–6 步（对应规定的 Level0 窗口） | 29,678 | 16,832 | 16,834 | 56.72% | 56.72% | 0.67 |
| 第 10–59 步（对应稳态窗口） | 494,640 | 115,305 | 280,960 | 23.31% | 56.80% | 3,313.10 |

从第 8 步起两腿的命中数分开。第 8 步 A/B 命中 5,534/5,611 次，第 9 步为 3,496/5,615 次，第 10 步为 2,299/5,612 次。第 4–6 步比两腿各自的 50 步稳态窗口早，因此该段 Free 只覆盖缓存仍有效的早期步骤。CANN v9.1.0 源码的 executor cache manager 是线程局部对象，默认每个 manager 容纳 10,000 项，legacy 哈希表满后会拒收新键。上述计数直接观测到 executor 查找结果，未记录每次插入拒绝的原因。[A 逐步证据](../evidence/cache-diag/per-producer/01-A/steps.csv) · [B 逐步证据](../evidence/cache-diag/per-producer/02-B/steps.csv)

### 复用发生在哪里

60 个完整的 `[batch_size, seq_len]` 全部互异。稳态窗口新增 165,655 次命中，其中 165,384 次（99.84%）是**同一步里再次查询同一算子键**，另 271 次的上一次查询在更早的步。两腿首次查见非零键的 miss 均为 107,602 次。按 batch size 1、2、3、4 分组，分别覆盖 10、9、16、15 个测量步，每步净增 3,309.3、3,323.2、3,312.4、3,310.3 次命中。[分组复算](cache-comparison.json)

| 稳态窗口算子 | A/B 每步查找 | A 每步命中 | B 每步命中 | 每步多命中 |
|---|---:|---:|---:|---:|
| `aclnnDivs` | 1,049 | 4 | 1,041 | 1,037 |
| `aclnnMean` | 1,033 | 0 | 1,031 | 1,031 |
| `aclnnAdd` | 1,035 | 1 | 1,031.2 | 1,030.2 |
| `aclnnInplaceCopy` | 3,218 | 1,091 | 1,100.44 | 9.44 |
| `aclnnEmbedding` | 1,027 | 1.08 | 1.76 | 0.68 |
| `aclnnEmbeddingDenseBackward` | 1,027 | 1.08 | 1.76 | 0.68 |

Divs、Mean、Add 合计占新增命中的 93.51%。命中路径从 `hit_cache` 获取 executor 并跳过显式重建描述符及调用 `GetWorkspaceSize` 的慢路径；非零 workspace 仍须取得缓冲。Embedding 前反向保留了绝大部分 miss。真实下发时间和每个 stream 上的工作序列需从独立 Level1 对照读取。[全部算子增量](operator-deltas.csv) · [A 算子计数](../evidence/cache-diag/per-producer/01-A/operators.csv) · [B 算子计数](../evidence/cache-diag/per-producer/02-B/operators.csv)

以训练 PID 对齐，主线程在稳态 50 步命中 115,020 → 221,380 次，另一 producer 线程命中 285 → 59,580 次。每个线程均按自己的 `(op, hash)` 定义已见键，重新解析原始审计 TSV 后核对每步、每个算子的 hit、miss 及不可用次数与在机记录一致。[A 线程明细](../evidence/cache-diag/per-producer/01-A/producer-steps.csv) · [B 线程明细](../evidence/cache-diag/per-producer/02-B/producer-steps.csv) · [线程局部 cache manager](https://gitcode.com/cann/opbase/blob/0e247b8c759c48ae42fda6c6f2bd5daab323be0f/src/nnopbase/common/utils/op_cache.cpp#L88)

### 同一训练进程第二次遇到同一形状

另用默认容量走完标准序列前 8 个形状，随后按原顺序再生成这 8 种形状的新输入。两遍各有 8,216 次 `aclnnEmbedding` 和 8,216 次 `aclnnEmbeddingDenseBackward` 查找。首次遇到这些完整形状时，两类算子各有 5 次命中、8,211 次 miss；第二遍两类算子均为 8,216 次命中、0 次 miss。对每一种已见形状，前向和反向的 1,027 次查找都复用了更早步骤的键。首次到第二次的显式 workspace 准备慢路径因此各少 8,211 次。此处测的是缓存返回结果和所选路径的次数，单次准备时长取 Level1。这 16 步独立于正式 50 步测速。[8+8 实际序列](../evidence/repeat/workspace/configs/dynamic_shapes_npu8.json) · [逐形状配对](../evidence/repeat/per-producer/01-A/repeat-shape-pairs.csv) · [按算子合计](../evidence/repeat/per-producer/01-A/repeat-op-totals.csv)

### Host 内存

诊断进程第 59 步末 RSS 为 2,200,136 / 2,743,048 KiB，B 多 530.19 MiB。第 0 步末至第 59 步末，A、B 的 RSS 分别增加 79.01、652.92 MiB。RSS 包含训练进程和审计库的全部常驻页，无法从该指标单列 executor 或推算其他形状分布的内存需求。[两腿 RSS](cache-comparison.json)

在本目录运行 `python compare_diagnostics.py ../evidence/cache-diag/per-producer` 即可重建 `cache-comparison.json` 和 `operator-deltas.csv`。程序校验两腿逐步形状、每个算子的查询总数，并核对各算子细项与每步统计。

逐线程复算使用 `recompute_archive.py <证据目录> <shape序列> --steps <60或16>`，从该目录中的 `raw-lookup-markers.tgz` 重建 `per-producer/` 并逐项核对原始 hit/miss。8+8 结果运行 `python compare_repeats.py ../evidence/repeat/per-producer/01-A` 重建。
