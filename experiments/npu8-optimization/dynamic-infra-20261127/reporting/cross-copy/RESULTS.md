# D04 跨源分配流连续 D2D 的动态 eager 配对

**尝试无优化路径。** 跨源流直接复制确实缩短了目标请求的 Host 准备与服务，独立复核的端到端方向反转，第一轮两序也均未降低三步绝对 Free。

训练两腿使用同一实验库，共同启用 #42 同分配流直接 D2D、01 overflow、TQ Mode 1、原生 ZeRO-2 分区和 norm。A 只标记满足跨分配流 guard 的复制，走原有 ACLNN 实现，B 则同时记录源块在当前流上的使用并直接提交复制。每次请求保持一个 ACL memcpy 和同物理流、同流内序号上的一项 SDMA。[训练开关](../../cross-copy.json) · [实际实现](../../cross_copy/from-stock.patch)

8 步训练语义门两腿的库 SHA-256 均为 `047f7d4f45f1e831472f3d5fa11077ebdd1f016fae74e74b268c37f50259d753`。两腿各 8 × 1,052 个梯度摘要、norm、BF16/FP32 权重及 AdamW 状态一致，loss 最大绝对差 3.815e−6。#2、#4、#6 各注入一种非有限值并跳步，此后恢复更新。由于该构建未导出训练计数器，激活取两步 Level1 的逐笔原生 marker。[语义门](../../evidence/cross-gate/comparison.json) · [激活与未标记队列项](../../analysis/cross-copy-l1-ab/copy_guard_evidence.json)

定向微测中，源在其他流分配、目的分配流匹配的 BF16/FP32 连续复制命中 #44；同流复制命中 #42。目的分配流不合格、跨 dtype、非连续、广播、共用 storage 均由原路径处理，部分重叠保留异常。另在 producer→consumer 等待后，复制尚未完成时立即释放源，并持续申请同尺寸块。显式 `record_stream` 时，A、B 的 `[64,64]` BF16 和 `[128,64]` FP32 结果都正确。跳过调用者显式保护时，A 的两例均出现复制完成前复用且目标值不符，B 均延迟复用且目标值正确。这个用例检验 B 新增的隐式保护，不能据此判定遵守跨流存活约定的普通 A 调用有错。[A 微测](../../evidence/cross44-micro/01-A/micro.json) · [B 微测](../../evidence/cross44-micro/02-B/micro.json)

Level1 两个实际形状各有 1,052 次跨源流复制命中。目标组的 producer 和 consumer p50/p95 均下降，目标 SDMA 前的全流空档合计分别减少 15.630、13.309 ms；ACL memcpy 自身 p95 均上升。第 5 步 ACL memcpy 最大值 21.50 → 143.14 µs，两步全步 Level1 Free 均升高。[逐步阶段、长尾与流内提交](../../analysis/cross-copy-l1-ab/RESULTS.md)

同步窗口的样本/s、token/s、同形状 Host 差值和每步 Level0 原始口径均可复算。八列使用第一轮同序的稳态与 Level0。独立复核轮 `cross-check` 只使用无 profiler，A/A 表示两次相同基线的重复。[完整计时与配对波动](measurement.md) · [各腿指标和命令来源](paired_results.json)

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 671.667 ms/步 | 668.477 ms/步 | +0.47% | 638.591 ms/步 | 92.43% | 757.029 ms/步 | 93.62% | −1.29% |
| 644.590 ms/步 | 599.690 ms/步 | +6.97% | 781.432 ms/步 | 93.88% | 786.968 ms/步 | 93.87% | +0.01% |

首行 A/B，次行 B/A。两序的标准 Free 分别增加 118.438、5.537 ms/步，反序 Free 占比相对变化仅 +0.00968%。正序三个形状的 Free 各自都增加；反序只有第 6 步增加，从 760.171 升至 853.517 ms。标准窗口内没有看到整步绝对 Free 缩短。[Level0 三步边界](profile_steps.csv) · [正序原始结果](../../evidence/cross-copy-l0-ab/status.json) · [反序原始结果](../../evidence/cross-copy-l0-ba/status.json)

额外 A/A 的两次相同基线从 649.705 到 684.452 ms/步，相差 34.747 ms（5.35%）。第二轮 A/B 694.429 → 613.964 ms/步（+11.59%），B/A 631.395 → 684.630 ms/步（−8.43%），两序相差 20.02 个百分点。复核轮 50 个相同步号中有 29 个的逐步 Host 墙时方向相反或相等，仅 13 个双序 B 均缩短、8 个均增加。四个跨处理同步窗口的相对变化范围为 −8.43% 至 +11.59%；单笔加速未形成可重复的整个训练窗口收益。[A/A 回执](../../evidence/cross-check-no-profiler-aa/status.json) · [复核正序](../../evidence/cross-check-no-profiler-ab/status.json) · [复核反序](../../evidence/cross-check-no-profiler-ba/status.json)

在实验根目录运行 `python -B reporting/recompute_recipe.py cross-copy` 可重建计时和波动表。训练 shape 覆盖 60 个合成形状，batch size 1–4、序列长度 12–32。
