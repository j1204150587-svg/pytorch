# D03 修订版直接构造 Meta 模板权重视图

两腿采用同一个 TorchNPU 包，关闭原 #43、#41 和队列复制回放，保留动态 eager、原 ZeRO-2 分区和 norm、01 overflow、TQ Mode 1。B 通过 `torch._utils._unflatten_dense_tensors` 的 TorchNPU 适配层识别连续 BF16 NPU flat 与大组普通 BF16 Meta 模板，由原生 `try_unflatten` 为每个模板直接构造 `as_strided` 视图。特定 autograd、tracing、dispatch mode 和布局条件回到原 helper。[候选开关](../../meta-views.json) · [适配函数](../../meta_unflatten/adapter.py) · [原生构造](../../meta_unflatten/meta_unflatten.cpp)

原 #43 在真实训练中未命中。显式指定 AutogradPrivateUse1 的对照能进入旧实现，实际大组传入 1,052 个 Meta 模板，触发另一条分派路径。D03 修订版引入的 sidecar SHA-256 为 `dc14f01851cf4f9bcf3cd8c5e693e3fbe8a6032fce0ec5f47be26d60b2fe2006`，两腿原有 `libtorch_npu.so` 均为 `36faab10edae336be136c02fdbb2ace3d3dcdb910ddfc97b8b2226e22f03adcc`。[旧实现实际分派](../../unflatten_probe/RESULTS.md) · [构建回执](../../provenance/meta/build-receipt.json)

八步语义门中 B 调用 13 次，大组命中 5 次（对应五个实际执行权重更新的步骤），其余 8 次回退。8 × 1,052 个梯度摘要、FP32/BF16 权重及 AdamW 状态一致；norm 最大差为零，loss 最大绝对差 7.629e−6。注入非有限值的第 2、4、6 步跳过更新，对应步骤的大组命中均为零。[训练语义门](../../evidence/meta-gate/comparison.json) · [逐步激活](../../evidence/meta-gate/02-B/rank000/native_activation.json)

定向实机微测检查 3 个 flat 起始位置，每次构造新输出，核对形状、stride、storage offset、共享 storage、值以及就地更新后的版本传播。B 共 6 次命中、11 次回退。空与混合模板、小列表、错误 dtype、非连续 flat、稀疏模板、inference、前向和反向 AD、dispatch mode 均覆盖。另一个进程在首次命中之后注册第三方 AutogradMeta 实现，下一次对 32 个模板走回退并调用该实现一次。[实机微测](../../evidence/meta-micro/02-B/micro.json) · [后注册实现](../../evidence/meta-override/01-B/micro.json)

Level1 第 4、5 步的权重回写大组分别为 19.712 → 4.506、25.191 → 5.526 ms。每步各省去 1,052 个 `narrow`、1,052 个 `slice` 和 1,052 个 `view`，1,052 个构造输出的 `as_strided` 仍然执行。第 4 步末项到下一步首项设备工作的空档为 33.782 → 20.413 ms；第 5 步在采样末尾缺少下一步边界。两腿同一实际 shape 的三条物理流工作序列及 ACL 提交数相同，减少的调用都发生在 Host。Level1 整步 Free 第 4、5 步分别为 851.255 → 733.811、803.351 → 752.134 ms，需与独立 Level0 三步配合解读。[分段归因和实际提交](../../analysis/meta-views-l1-ab/RESULTS.md) · [原始 Level1 结果](../../evidence/meta-views-l1-ab/status.json)
