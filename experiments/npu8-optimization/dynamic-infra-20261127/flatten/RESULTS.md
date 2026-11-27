# #41 整段 flatten 在默认动态 eager 中未触发

使用含 #41 的实际 PTA 库，`NPU_EXPERIMENT_FLAT_ND=1`。加载库 SHA-256 为 `36faab10edae336be136c02fdbb2ace3d3dcdb910ddfc97b8b2226e22f03adcc`，库中含 `NPUDenseFlattenStorageCopy`。前 8 步、8 个实际 shape 中共记录 16 次原生 flatten，快路径命中 0 次。[实机逐次元数据和计数](../evidence/flatten-probe.jsonl)

| 训练入口 | 调用次数 | 输入布局 | 实际执行 |
|---|---:|---|---|
| 梯度桶 `allreduce_bucket` | 8 | 每次只有 1 个 tensor | 单张量分支直接展平，0 次 cat |
| 优化器 `flatten_dense_tensors_aligned` | 8 | 每次 1,052 个连续 BF16 分片，共享同一 ND storage；按调用列表顺序有 1,051 个位置偏离预期连续 offset | 每次仍执行 1 次 cat |

每步第二个分片均从 4,310,848 元素处开始，紧接第一个分片所要求的 offset 为 4,380,544。共享底层分配成立，原列表的物理顺序不满足整段 clone 的必要条件。按该列表做一次线性复制会改变梯度排列。此布局下没有 #41 的性能贡献可测。[守卫实现与复现](README.md) · [复算程序](summarize_flat.py)
