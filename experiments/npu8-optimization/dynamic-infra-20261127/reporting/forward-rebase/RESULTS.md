# D05 复用前向 embedding 的索引与权重描述符

两腿使用相同 op-plugin 构建，均为动态 eager、01 分块 overflow、原 ZeRO-2 梯度分区和 norm、Task Queue Mode 1，仅切换 `NPU_DYNAMIC_REBASE_FORWARD`。合格的侧表权重为 BF16 连续 `[64,64]`，索引为三维 int64 视图，末维宽度为 4。对两者同时按 512 字节移动描述符基址，并保留可达范围和余数 offset，由相同逻辑生成缓存键和 CANN 描述符。[配对配置](../../forward-rebase.json) · [视图守卫和规范化](../../rebase_forward/index_scope.inc)

在八个形状上，训练 B 的 8,192 次侧表调用命中，另外 24 次前向 embedding 回退，规范化键使用 16,384 次，实际规范化描述符构造 288 次。两腿库 SHA-256 均为 `5d9c217c544621fca4e1da5d018b1fb9fd1b34705730a3fe500bdc51726e2b7f`。每步 1,052 组梯度摘要、FP32/BF16 权重及 AdamW 状态一致，梯度 norm 最大差为零、loss 最大绝对差 3.815e−6。第 2、4、6 步注入 NaN、+Inf、−Inf 后按预期跳步，随后恢复更新。[训练语义回执](../../evidence/forward-gate/comparison.json)

定向微测跨四个不同形状和八个表位置，逐笔比对 80 份前向输出摘要。第五轮复访 `[1,19]`。B 计数 80 次命中，另有一次二维索引回退，规范化键使用 160 次、描述符构造 24 次。[A 微测](../../evidence/forward-micro/01-A/micro.json) · [B 微测](../../evidence/forward-micro/02-B/micro.json)

无 profiler 的两个独立配对均缩短 50 步同步窗口。正序 681.861 → 593.436 ms/步（+12.97%），反序 714.650 → 615.996 ms/步（+13.80%），降幅相差 0.84 个百分点。两序分别节省 88.424、98.654 ms/步。50 步实际工作量均为 136 个样本、2,994 个 token。[八列、实际吞吐、各形状 Free/Stage/Computing 与波动](measurement.md) · [A/B 原始命令和结果](../../evidence/forward-rebase-no-profiler-ab/status.json) · [B/A 原始命令和结果](../../evidence/forward-rebase-no-profiler-ba/status.json)

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 681.861 ms/步 | 593.436 ms/步 | +12.97% | 789.730 ms/步 | 93.74% | 733.680 ms/步 | 93.32% | +0.45% |
| 714.650 ms/步 | 615.996 ms/步 | +13.80% | 745.266 ms/步 | 93.54% | 759.789 ms/步 | 93.36% | +0.20% |

正序 Free 少 56.050 ms/步，反序多 14.523 ms/步。反序第 4、5 步分别少 41.973、17.455 ms，第 6 步多 102.998 ms，导致三步绝对 Free 方向相反。两序的 ΣFree/ΣStage 都略降，反序的 Computing 同时从 51.267 升到 53.900 ms/步。本轮支持端到端改善和 Free 占比小幅下降，绝对 Free 缩短未在双序复现。[各步及 CPU 边界](profile_steps.csv) · [正序原始结果](../../evidence/forward-rebase-l0-ab/status.json) · [反序原始结果](../../evidence/forward-rebase-l0-ba/status.json)

Level1 第 4、5 步各有 1,027 个前向 embedding 请求。每步 1,026 → 18 次可见的 GetWorkspaceSize；producer 准备 p50 为 62.98 → 16.95、64.60 → 18.65 µs，consumer 服务 p50 为 78.52 → 23.36、79.90 → 24.36 µs。相同 ATen 调用内的 ACL API 序列、物理 stream、设备工作任务流内位置和数量一致。每步有 1,024 个侧表的首个 kernel 由 Slice 改为 AsStrided。目标 kernel 前的全流空档各减少 48.917、32.224 ms。单次 ACL launch 的 p95 升高，观察到的单笔优势位于 Host 描述符与 executor 准备等环节。[逐笔与逐流核验](../../analysis/forward-rebase-l1-ab/RESULTS.md)

汇总复算命令为 `python -B reporting/recompute_recipe.py forward-rebase`。原始 60 个合成形状覆盖 batch size 1–4，序列长度 12–32。[完整配对索引](paired_results.json)
