# 动态 eager 的 ZeRO-2 统计与侧表梯度候选

既有 07 梯度打包和分块 norm 的动态 Level0 B 腿在 #4–6 有 322.887 ms/步官方 Free，约 50.033 ms/步的非 Free。固定这部分工作量，Free 需低于约 116.74 ms/步才达到 70%。optimizer 全阶段每步仅有 43.24–44.03 ms 所有流上没有计算或搬运，内部 norm 占 9.37–9.50 ms。同时前向和反向各有 117.66–120.34、161.67–166.95 ms 这样的空档。Host scope 的未覆盖时间与官方 Free 取自不同的时间边界。[原始 Level0](../../dynamic-shapes-20261127/evidence/both-l0-ab/02-B/step_trace_time.csv) · [分阶段原始分析](../../dynamic-shapes-20261127/evidence/both-l0-ab/02-B/trace_analysis.json)

## 使用真实梯度 flat 做 eager 统计

07 在单 rank 每一步形成新的连续 BF16 梯度，交给 ZeRO 的 1,052 个分片依次引用其中的区间。01 的检查逐片查格式、拼接后每步扫描 17 个块。07 的 norm 再逐片查格式，把同形参数堆叠成块做逐行 L2。实验接入 ZeRO 的两个 Python 实例方法，在 NPU 上执行 eager ATen，保留同一份梯度、ZeRO 裁剪、overflow 后的跳步和 AdamW 更新。每次调用检查发布方、分片的 `_base`、偏移、形状、连续性、dtype 及整组 ND 格式，守卫失败时交还 01/07。已知边界是外部在发布后对同一分片对象调用 `set_` 改绑 storage，却仍保留 `_base`、shape/offset/stride；07 的当前训练循环不做该操作。其他发布方需要直接验证实际 StorageImpl 身份。

- `overflow` 用 `~torch.isfinite(flat).all()` 检查当前完整梯度缓冲。有限 BF16 值即使相加溢出也仍属于有限输入。指标为 17 块降为 1 个逻辑扫描、大量逐片格式检查和 Cat 是否消失、实际 ACL 与设备任务减少多少。初次取证不包含硬件溢出标志的模式。
- `norm` 严格使用原 07 的参数顺序、相邻同形分组、元素和张量上限及 double 转换。每个块在 flat 上取视图，用原行规约和最终平方求和；仍调用原 allreduce、平方根和非有限处理。检查 `stack` 的实际设备提交是否减少，并比对真实梯度 norm、裁剪和更新。
- `both` 同时启用这两个入口，仅可用于与标明的前置状态的组合配对。统计和输入只读取实际新发布的梯度；没有缓存上一训练步的 norm 或 overflow 判断。

```bash
python stage_flat_stats.py <已安装07和01的隔离训练workspace>
python stats_gate.py --device npu --output <基于当前镜像的专项门.json>
```

两腿均设置 `RECSYS_DYNAMIC_BOTH=1`、`RECSYS_ZERO2_BATCH_OVERFLOW=1`，A 的 `RECSYS_DYNAMIC_FLAT_STATS=off`。B 可单独选择 `overflow` 或 `norm`。报告文件为 rank 目录的 `flat_stats_counts.json`，包含实际启用及回退次数。gate 所用的 8 步中有 5 次有效更新及 3 次注入跳步；由此分别预期 8 次 overflow 和 5 次 norm。[实现](flat_stats.py) · [独立专项门](stats_gate.py) · [本机 CPU 数学对照](stats-gate-cpu.json)

## 侧表反向的单 kernel 备选

已有动态 grouped 版本通过一次 `F.embedding` 和 `mean` 自然保留一次 dense embedding backward，`CatBackward` 把整组梯度切回原参数。本备选保持相同的 grouped 前向，替换该算子的 backward，使一个 Triton program 处理一张表。程序按 token 分块，统计该表每个局部 ID 在四个特征中出现的次数，乘以原 BF16 的 `grad/4` 后累入 FP32，每张表输出 64×64 个 BF16 梯度元素。输入有非有限值时按匹配的行单独处理，避免无关行的零系数乘以 Inf 或 NaN。候选仍需验证矩阵乘积累加顺序、相消及 BF16 舍入。

微测从原始逐表 NPU embedding/mean 的反向取得预期梯度；检查尺寸 `[1,19]`、`[2,14]`、`[2,22]`、`[4,31]`，跨 token 和表内的重复 ID、边界行、相消及非有限。有限值、无穷和带符号零按位核对，NaN 按分类核对。与原梯度存在差异的 kernel 不进入 HSTU 性能配对。

```bash
python dense_gate.py --tables 1024 --output <dense-gate.json>
```

`GroupedSideGrad.apply(weight,ids,offsets)` 接受连续 BF16 `[T*64,64]` 权重、int64 `[B,L,T,4]` 的合法局部索引，以及现有 grouped 前向的偏移。独立比较 kernel 时可直接调用 `grouped_dense_backward(ids,grad)`，输出为 `[T,64,64]`。以相同 grouped 前向、相同 ZeRO 梯度所有权和相同前置项为 A，单切后向实现。证伪条件包含反向梯度差异、消掉提交却加重实际设备耗时、双序 Free 或端到端无改善。[候选](dense_grad.py) · [Triton 实现](dense_grad_kernel.py) · [逐表对照](dense_gate.py)
