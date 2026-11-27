# #41 在动态 eager 的触发条件

DeepSpeed v0.19.0 的 ZeRO-2 初始化把 `self.flatten` 绑定到 `_flatten_dense_tensors`。在单 rank、单参数组的优化器步中，该 rank 同时是最后一个分区。ZeRO 将 `averaged_gradients[i]` 交给 `flatten_dense_tensors_aligned`，按分区大小补齐后展平，再转 FP32 供裁剪和 AdamW 使用。本探针还观察同一实例在训练循环中调用 `self.flatten` 的其他位置。[DeepSpeed 绑定](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.0/deepspeed/runtime/zero/stage_1_and_2.py#L210) · [待展平的梯度列表](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.0/deepspeed/runtime/zero/stage_1_and_2.py#L2182)

历史 #41 在 `NPUNativeFunctions::flatten_dense_tensors` 判断输入至少有两项，首项当前格式和原始格式均为 ND。同 dtype、同 device、连续、非空、无需梯度、无前向 AD、无命名或特殊布局的输入必须共用一个 StorageImpl，各片按原顺序无缝衔接且在容量之内。满足条件后在 `NPUDenseFlattenStorageCopy` 内对整段做一次 clone，生成独立输出；否则沿用逐张量展平和 cat。[原生守卫与 marker](../provenance/prior-materials.md)

在研究目录执行 `python3 -u flatten/run_flat_probe.py`。脚本使用独立 runner 副本和宿主共享锁，在现有 views 版本的 PTA 中开启 #41，关闭 #40 与 #43。HSTU 走默认动态 eager、01 开启、单 rank、GAS 1、TQ1，运行标准序列的前 8 步。每次调用保留原 tensor 列表并记录存储数、偏移衔接、元素个数、dtype、梯度/前向 AD、叶子布局以及第一个违反守卫的参数位置。CPU profiler 仅包住原 `flatten` 调用，记录 ATen 入口、原生命中和 cat。全部元数据及真实库 SHA-256 在 `01-A/rank000/flatten-probe.jsonl`。诊断步时不可用于 A/B。若有命中，独立性能配对决定是否归档；若每步均可见实际 flatten、每次输入违反至少一项硬条件，即可说明当前训练布局中的不可用原因。
