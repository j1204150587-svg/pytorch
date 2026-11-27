# D02 规范化 embedding 反向索引视图的动态 eager 配对

开关位于 op-plugin 的 `embedding_dense_backward`、`add_param_to_buf` 和 `ConvertType`。B 对符合守卫的三维 int64 只读索引视图，以 512 字节为单位移动描述符基址，保留剩余 offset 并同步规范化缓存键。A、B 运行同一个独立构建，只改变 `NPU_DYNAMIC_REBASE_INDICES=0/1`。两腿均为原生 ZeRO-2 分区和 norm、01 分块 overflow、Task Queue Mode 1。[源码改动](../../rebase/apply_rebase.py) · [配对开关与公共构建位置](../../index-rebase.json)

8 个形状的注入与恢复测试通过。两腿加载库 SHA-256 均为 `4f99aa6b3d4017b37402af361e0dea364fb327910a2995135e8cb370482d122b`。B 共 8,192 个索引命中、24 个回退、8,192 次用到规范化缓存键、128 次进入规范化描述符构造，A 四项计数均为零。逐步 1,052 个梯度、BF16/FP32 权重、AdamW 状态匹配，#2、#4、#6 跳步，第 7 步恢复更新，最大 loss 绝对差 3.815e−6、norm 差为零。[实机语义和激活回执](../../evidence/rebase-gate/comparison.json)

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 606.854 ms/步 | 495.278 ms/步 | +18.39% | 589.394 ms/步 | 91.87% | 634.877 ms/步 | 92.55% | −0.73% |
| 530.369 ms/步 | 477.281 ms/步 | +10.01% | 693.014 ms/步 | 92.95% | 660.062 ms/步 | 92.70% | +0.27% |

首行按 A/B 执行，第二行按 B/A 执行。60 步训练按同一条实际形状序列生成输入，无 profiler 的首尾同步窗口覆盖 #10–59，共 136 个样本、2,994 个 token。两对端到端分别少 111.576、53.088 ms/步，降幅相差 8.38 个百分点，中位降幅 14.20%。[正序稳态](../../evidence/index-rebase-no-profiler-ab/status.json) · [反序稳态](../../evidence/index-rebase-no-profiler-ba/status.json)

| 顺序 | A 样本/s | B 样本/s | A token/s | B token/s | A / B Stage（ms/步） | A / B Computing（ms/步） |
|---|---:|---:|---:|---:|---:|---:|
| A/B | 4.4821 | 5.4919 | 98.6728 | 120.9017 | 641.532 / 686.002 | 51.976 / 50.961 |
| B/A | 5.1285 | 5.6990 | 112.9025 | 125.4607 | 745.564 / 712.043 | 52.409 / 51.819 |

**标准 Level0 Free 的改善未复现。** 正序 B 多 45.483 ms/步 Free，占比上升 0.675 个百分点。反序 B 少 32.952 ms/步 Free，占比下降 0.252 个百分点。分母均取该腿 CSV 的三个 Stage 之和。各步 Free 如下。[正序 Level0](../../evidence/index-rebase-l0-ab/status.json) · [反序 Level0](../../evidence/index-rebase-l0-ba/status.json)

| 步 | 实际 [B,L] | 正序 A / B Free（ms） | 反序 A / B Free（ms） |
|---:|---|---:|---:|
| 4 | [1,19] | 569.500 / 641.253 | 685.038 / 682.253 |
| 5 | [2,14] | 571.599 / 651.970 | 702.905 / 659.083 |
| 6 | [2,22] | 627.082 / 611.408 | 691.099 / 638.848 |

最后活跃步 #6 四腿 CPU `ProfilerStep` 墙时比官方 Stage 长 30.33–39.93 ms。正式表格取完整 #4–6 官方 CSV。独立 Level1 正序的 #4、#5 观察到 Free 为 628.435 → 569.638 ms/步，该窗口仅用于调用归因。[各步 CPU 与设备边界](profile_steps.csv)

每步仍有 1,027 个 embedding 反向请求、6,159 次该算子的 ACL kernel launch。Level1 的 ATen 入口到 enqueue 开始 p50 在 #4 为 76.04 → 15.36 µs、#5 为 74.45 → 15.21 µs；consumer 服务 p50 为 158.18 → 59.28 µs、150.15 → 68.77 µs。设备内部每步 1,024 个侧表请求各将一个 `Slice` 换成 `AsStrided`，两腿逐流任务数量对齐。单个 ACL launch p95 在两个步骤均上升，且 #5 B 的可见队列等待 p95 达到 17.482 ms。当前数据支持整步缩短、描述符和 executor 准备负担下降，并包含设备内部实现的变化。[同算子、逐流工作与分位数复核](../../analysis/index-rebase-l1-ab/RESULTS.md)

在本轮实验目录运行 `python reporting/recompute_recipe.py index-rebase` 即可重建各表。本轮覆盖 batch size 1–4、序列长度 12–32 的 60 个合成形状。[配对 JSON](paired_results.json) · [同步窗口](synchronized_windows.csv)
