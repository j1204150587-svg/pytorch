# 07 直接接收稠密侧表梯度

v2 在侧表 Eager backward 得到 `[65536,64]` 的 BF16 dW，再复制 8 MiB 到固定缓冲，向 1,024 个 Parameter.grad 写入独立视图，原 07 读取并清除这些 grad。本项以 v2 与 partition_pack 为共同前置，B 单独启用 `NPU_DYNAMIC_DIRECT_SIDE=1`。输入拓扑、权重和 dW 算法不变；实验改变的是 07 接收和发布梯度的过程。[实现](side_publish.py) · [接入](stage.py)

在启动时校验真实侧表参数区间、最终 flat 的 dtype/设备/计算流、全部 1,052 个目标视图的偏移和布局。每步在消费 dW 之前核对活动代次、当前组、私有输出、分配及 stream 所有权。只有通过守卫，才将普通 3 个前置梯度、单个实际侧表 dW、普通 25 个后置梯度，按原 ZeRO 顺序合为 29 个输入。已有 pack 接收这些 segment，执行一次 `cat(out=flat)`，向 ZeRO 发布 1,052 个已建好的视图，清除 28 个普通参数梯度。该步侧表参数的 `.grad` 保持为空，避免设置后立即读取和清除。最终视图仍有原来的表别名、边界、顺序和元素，可逐张进行梯度注入与 overflow 判定。整个模型的 Checkpoint 和 AdamW 参数身份在本项中保持原状。

若启用了 flat_stats，消费端只为当前已发布的私有输出跳过逐片验证。视图列表结构发生变化、flat 版本发生变化、引用或所有权不一致时走原检查。合法的原位数值注入会改变张量版本，仍由原有验证和非有限值检测处理。该证明限于受控训练入口和其已核验的 ZeRO 消费者，外部直接对张量使用 `.data` 绕过版本计数不在该实验契约内。

按先后顺序将 side_tables、side_slab、partition_hints、partition_pack、可选 flat_stats 和 side_publish 应用到新的 staged workspace，再启动原 runner。每步命中时 `side_slab_counts.json` 应显示 `direct_partitions` 加 1、`direct_final_views` 加 1,052、`ordinary_gradients_checked` 加 28。v2 的 `gradient_buffer_copies` 和 `published_table_views` 仅在 v2 回退产生。整步和 Level0 必须使用独立配对，是否少了一次真实 SDMA 需要 Level1 逐流核对。

CPU 使用真实 staged 07 分区及 pack 代码，单 rank ZeRO fixture 构造真实连续 BF16 存储。在前三个形状中全部 1,052 个发布梯度与 A 逐位一致，每步只清除 28 个普通参数的梯度；通过已发布视图注入 Inf 后，底层 flat 可见，快速验证退回原始检查。第四个形状人为更改视图列表的结构版本，v2 回退仍发布正确梯度并清除全部 1,052 项。[CPU 原始结果](cpu-gate.json) · [CPU 验证入口](cpu_gate.py)
