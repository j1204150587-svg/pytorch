# 动态 eager 的 07 梯度分区

原 07 从每个参数的 `grad` 构造一个扁平视图，做一次 `cat`，随后按参数拆分并再建输出视图。候选仍从同一组参数按原序取梯度，single rank、梯度累积 1、BF16、无 padding、单参数组和通信 dtype 由原 07 和新的分区入口共同校验。两腿共同启用 `partition_hints` 中的 #09，独立开关为 `NPU_DYNAMIC_PARTITION_PACK`。[训练入口](stage.py) · [分区实现](partition_pack.py)

每一步读取真实梯度的 StorageImpl 和 storage offset。相同 storage 内相邻的逻辑元素段可变成一份较长的一维输入；遇到不同 storage、反向顺序、间隙或重叠即断开。对 NPU 多张量合并只接受 ND storage。各段按参数顺序一次 `cat(out=flat)` 写入私有缓冲，首次准备的 1,052 份分片视图随同步的 ZeRO 消费循环复用。原 07 的 `averaged_gradients`、`_recsys_local_flat_gradients` 和参数梯度清理仍按原序发布。读写使用同一训练流；跨生产流、源与目标 storage 重合及不合格布局走原 07，消费者流变化会使这次试验失败。每步只有正常的 gradient cat，不插入设备同步、事件、额外 memcpy 或填充算子。

计数回执 `eager_partition_pack.json` 保存候选命中和回退、实际 cat 输入个数范围、合并掉的相邻边界及复用输出视图的步数。父级语义门的非有限注入作用在真正的分区视图，核验 loss、分参数梯度、norm、跳步、FP32/BF16 权重和 AdamW。Level1 用 `EagerPartitionPackCat` 匹配实际 kernel 与 SDMA，再从独立无 profiler 窗口和 Level0 三步判定整步目标。不得从 Python 的一次 cat 推断设备一定只有一个任务。

离线 BF16 CPU 用例在非零 offset 的共享 storage 放 1,024 张侧表，首尾再加独立梯度，cat 输入从 1,026 个缩到 3 个。逐参数值、顺序、别名、第二次输出 view identity、无穷值传播以及重叠、换 shape 和换 dtype 的回退均已检查。此用例只覆盖分区语义。[CPU 检查程序](cpu_gate.py) · [实测回执](cpu_gate.json)
