# D03 权重恢复的 NPU flat 与 Meta 模板

旧 #43 库在真实调用上未触发。动态训练连续八步中，`_update_model_bit16_weights` 每步接受一个 NPU ND 格式的 BF16 `[4380544]` flat、1,052 个普通 BF16 Meta 模板。各模板均不含梯度或前向 tangent，长度总和恰好填满 flat。八步另一处 `allreduce_and_copy_with_multiple_ranks` 只有一个 NPU 模板。两处开着全局 grad mode。在指定 wheel 中，`aten::unflatten_dense_tensors` 的 Meta、AutogradMeta 分派均来自 CompositeImplicitAutograd math kernel；AutogradPrivateUse1 行是旧 #43 的显式 kernel。上述观测独立于 HSTU 的八组实际输入 shape。[实机输入与完整分派表](../evidence/unflatten-probe.jsonl) · [原版 #43 Level1](../analysis/unflatten-l1-ab/RESULTS.md)

`adapter.py` 随 PTA 实验包加载，只有 `NPU_DYNAMIC_META_UNFLATTEN=1` 才替换 `torch._utils._unflatten_dense_tensors` 的入口。先检查 flat 为普通 BF16、格式 ND 的 1-D 连续 NPU 张量，模板为至少 32 个原生 Tensor；特殊分派模式或显式 Meta / AutogradMeta 实现交回原函数。自有 CompositeImplicitAutograd 操作 `npu_dynamic_meta_views::try_unflatten` 再逐个检查 flat 的别名与 AD 属性以及每个 Meta 模板。生成每个输出时直接以正确的 storage offset 和连续 strides 调用一次 `as_strided`，保持独立 view identity、底层存储和版本传播。空模板、元素越界、AD、非普通模板等输入返回原 helper，保留原输出或错误语义。[Python 入口](adapter.py) · [原生守卫与视图构造](meta_unflatten.cpp)

独立构建只增加一个 sidecar 动态库和 Python 入口，复制的原有 `libtorch_npu.so` 与基线同 SHA。A、B 使用同一包，将旧的 `NPU_EXPERIMENT_UNFLATTEN_VIEWS` 都置 0。将下面的输出路径替换为新的共享实验目录，在相同容器与逻辑卡映射下分两次运行。

```bash
python build.py --base-package /artifacts/infra-discovery-20261127-views/package --output /artifacts/dynamic-infra-20261127/native-meta-v2
export PYTHONPATH=/artifacts/dynamic-infra-20261127/native-meta-v2/package:${PYTHONPATH}
export NPU_EXPERIMENT_UNFLATTEN_VIEWS=0
NPU_DYNAMIC_META_UNFLATTEN=0 python micro_gate.py /artifacts/dynamic-infra-20261127/meta-gate-A.json
NPU_DYNAMIC_META_UNFLATTEN=1 python micro_gate.py /artifacts/dynamic-infra-20261127/meta-gate-B.json
```

机制门检查每步大组 `NPUDynamicMetaUnflattenViews` marker 和父 scope，独立计数 `calls/eligible/fallback`。预计每个合格大组保留 1,052 个 `as_strided`，省去各 1,052 个 `narrow/slice/view`；1 个 NPU 模板的小组走原始分派。该路径不需要设备 kernel 或 SDMA，收益指标为权重恢复期间的墙钟、全流已无工作的时间、标准三步 Free 和双序端到端。当真实大组未命中、值/梯度/跳步不一致、设备实际提交变化或双序不能复现收益时，分别记录该项证据。
