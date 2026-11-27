# D03 修订版直接构造 Meta 模板权重视图

**尝试无优化路径。** 新实现确实将两个实际形状上的权重视图构造分别缩短 15.206、19.665 ms，但四个无 profiler 同步窗口有三个回退。追加 A/B、B/A 两序分别慢 8.99%、2.57%；标准 Level0 的全步 Free 两序方向相反。

两腿采用同一个 TorchNPU 包，关闭原 #43、#41 和队列复制回放，保留动态 eager、原 ZeRO-2 分区和 norm、01 overflow、TQ Mode 1。B 通过 `torch._utils._unflatten_dense_tensors` 的 TorchNPU 适配层识别连续 BF16 NPU flat 与大组普通 BF16 Meta 模板，由原生 `try_unflatten` 为每个模板直接构造 `as_strided` 视图。特定 autograd、tracing、dispatch mode 和布局条件回到原 helper。[候选开关](../../meta-views.json) · [适配函数](../../meta_unflatten/adapter.py) · [原生构造](../../meta_unflatten/meta_unflatten.cpp)

原 #43 在真实训练中未命中。显式指定 AutogradPrivateUse1 的对照能进入旧实现，实际大组传入 1,052 个 Meta 模板，触发另一条分派路径。D03 修订版引入的 sidecar SHA-256 为 `dc14f01851cf4f9bcf3cd8c5e693e3fbe8a6032fce0ec5f47be26d60b2fe2006`，两腿原有 `libtorch_npu.so` 均为 `36faab10edae336be136c02fdbb2ace3d3dcdb910ddfc97b8b2226e22f03adcc`。[旧实现实际分派](../../unflatten_probe/RESULTS.md) · [构建回执](../../provenance/meta/build-receipt.json)

八步语义门中 B 调用 13 次，大组命中 5 次（对应五个实际执行权重更新的步骤），其余 8 次回退。8 × 1,052 个梯度摘要、FP32/BF16 权重及 AdamW 状态一致；norm 最大差为零，loss 最大绝对差 7.629e−6。注入非有限值的第 2、4、6 步跳过更新，对应步骤的大组命中均为零。[训练语义门](../../evidence/meta-gate/comparison.json) · [逐步激活](../../evidence/meta-gate/02-B/rank000/native_activation.json)

定向实机微测检查 3 个 flat 起始位置，每次构造新输出，核对形状、stride、storage offset、共享 storage、值以及就地更新后的版本传播。B 共 6 次命中、11 次回退。空与混合模板、小列表、错误 dtype、非连续 flat、稀疏模板、inference、前向和反向 AD、dispatch mode 均覆盖。另一个进程在首次命中之后注册第三方 AutogradMeta 实现，下一次对 32 个模板走回退并调用该实现一次。[实机微测](../../evidence/meta-micro/02-B/micro.json) · [后注册实现](../../evidence/meta-override/01-B/micro.json)

Level1 第 4、5 步的权重回写大组分别为 19.712 → 4.506、25.191 → 5.526 ms。每步各省去 1,052 个 `narrow`、1,052 个 `slice` 和 1,052 个 `view`，1,052 个构造输出的 `as_strided` 仍然执行。第 4 步末项到下一步首项设备工作的空档为 33.782 → 20.413 ms；第 5 步在采样末尾缺少下一步边界。两腿同一实际 shape 的三条物理流工作序列及 ACL 提交数相同，减少的调用都发生在 Host。Level1 整步 Free 第 4、5 步分别为 851.255 → 733.811、803.351 → 752.134 ms，需与独立 Level0 三步配合解读。[分段归因和实际提交](../../analysis/meta-views-l1-ab/RESULTS.md) · [原始 Level1 结果](../../evidence/meta-views-l1-ab/status.json)

首轮 A/B 的 50 步同步窗口为 744.559 → 726.741 ms/步（+2.39%），B/A 为 576.542 → 633.970 ms/步（−9.96%）。正反序相差 12.35 个百分点。相同步号的 50 个 Host 单步墙时中，B 双序均缩短 2 次，均变慢 18 次，方向不同或相等 30 次。原 B/A 的第一腿在开始前卡况检查失败，原 A/A 的第一腿在结束后卡况检查失败，两次会话均为退出码 12、完成腿数零。恢复空闲后从整对首腿重新执行，计时只取通过所有前后检查的会话。[两次中断记录](../../provenance/interrupted-pairs.json) · [有效 A/B](../../evidence/meta-views-no-profiler-ab/status.json) · [有效 B/A](../../evidence/meta-views-no-profiler-ba/status.json) · [同步吞吐与配对波动](measurement.md)

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 744.559 ms/步 | 726.741 ms/步 | +2.39% | 634.296 ms/步 | 92.45% | 774.264 ms/步 | 93.66% | −1.32% |
| 576.542 ms/步 | 633.970 ms/步 | −9.96% | 921.131 ms/步 | 94.69% | 609.444 ms/步 | 92.18% | +2.65% |

正序、反序依次列出。Level0 正序三个形状的 Free 都升高，平均增加 139.968 ms/步；反序三个形状都下降，平均减少 311.687 ms/步。正反序的 A 腿 Free 自身相差 286.835 ms/步，远超 Level1 实测的 15–20 ms 大组视图构造节省。两序标准窗口未确认可重复的整步 Free 收益。[逐步 Free/Stage/Computing](profile_steps.csv) · [正序 Level0](../../evidence/meta-views-l0-ab/status.json) · [反序 Level0](../../evidence/meta-views-l0-ba/status.json)

独立复核在读取首对 B 的结果前已安排。A/A 为 537.435 → 540.147 ms/步，第二腿慢 2.712 ms（0.50%）。A/B 为 557.044 → 607.108 ms/步（−8.99%），吞吐 4.8829 → 4.4803 样本/s、107.496 → 98.632 token/s。B/A 为 540.732 → 554.636 ms/步（−2.57%），吞吐 5.0302 → 4.9041 样本/s、110.739 → 107.963 token/s。复核轮比对 50 个相同步号，B 的 Host 单步墙时有 39 个在双序均增加，无一在双序均缩短，其余 11 个方向不同或相等。A/A 只描述这两次相邻基线；四个跨处理窗口的相对改善范围为 −9.96% 至 +2.39%，中位数 −5.78%。现有证据支持局部 view 构造和第 4 步末段设备空档缩短，整段训练不能复现收益。[A/A 回执](../../evidence/meta-check-no-profiler-aa/status.json) · [追加 A/B](../../evidence/meta-check-no-profiler-ab/status.json) · [追加 B/A](../../evidence/meta-check-no-profiler-ba/status.json)

在实验根目录运行 `python -B reporting/recompute_recipe.py meta-views` 可生成八列、逐步 CPU 与设备指标和各轮波动。无 profiler 各腿依次执行相同的 60 个合成形状，范围为 batch size 1–4、序列长度 12–32。第 10–59 步合计 136 个样本、2,994 个 token。Level0、Level1 各腿使用相同输入序列的前八步。[八列、吞吐和逐步窗口](measurement.md) · [同步窗口 CSV](synchronized_windows.csv) · [各腿完整计算](paired_results.json)
