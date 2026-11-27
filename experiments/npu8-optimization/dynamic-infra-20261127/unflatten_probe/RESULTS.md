# D03 原有 NPU unflatten 实现在权重回写处未被调用

开关 `NPU_EXPERIMENT_UNFLATTEN_VIEWS=1` 生效，实际加载的库 SHA-256 为 `36faab10edae336be136c02fdbb2ace3d3dcdb910ddfc97b8b2226e22f03adcc`。PyTorch 注册表将 `AutogradPrivateUse1` 指向候选的 `UnflattenViewsExperiment.cpp:71`，将 `AutogradMeta` 列为上游 CompositeImplicitAutograd math kernel。标准序列前 8 步的 16 次真实 unflatten 均进入 ATen，原生候选 marker 命中为 0。[原始 8 步回执及注册表](../evidence/unflatten-probe.jsonl)

| 训练入口 | 实际输入 | 每次可见的 narrow / slice / as_strided / view |
|---|---|---:|
| `_update_model_bit16_weights`，8 次 | 扁平 NPU BF16 权重 + 1,052 个 BF16 Meta 模板 | 各 1,052 |
| `allreduce_and_copy_with_multiple_ranks`，8 次 | 扁平 NPU BF16 梯度 + 1 个 NPU 模板 | 各 1 |

权重回写的 flat 为 `npu:0` 上的普通 strided ND 张量，形状 `[4380544]`、stride 1、offset 0，无反向或前向梯度、无 view base。1,052 个模板的张量类型都是普通 `torch.Tensor`，设备为 `meta`，dispatch keys 含 `Meta, ADInplaceOrView, AutogradMeta`。模板元素之和恰为 4,380,544。DeepSpeed 在初始化时显式用 `device="meta"` 构造这批模板，更新只需要其中的尺寸。[模板构造](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.0/deepspeed/runtime/zero/stage_1_and_2.py#L413) · [权重回写入口](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.0/deepspeed/runtime/zero/stage_1_and_2.py#L757)

第 7 步在原训练调用返回后，用同一组真实参数单独检查派发。临时替换已经注册的 `AutogradPrivateUse1`，显式指定该键时命中探针 1 次；常规调用命中 0 次，随后核对注册表完整恢复。权重回写的第一个失效点是正常调用绕过该 NPU 注册。即便显式路由到原实现，原守卫仍要求模板和 flat 同设备，并以 `isTensorSubclassLike` 排除 Meta；梯度桶调用则只有一个模板，低于原守卫的 32 项门槛。D03 的原实现由此不具备默认动态 eager 的激活条件。[探针](inspect_unflatten.py) · [原守卫与注册](../provenance/unflatten/unflatten.patch)
