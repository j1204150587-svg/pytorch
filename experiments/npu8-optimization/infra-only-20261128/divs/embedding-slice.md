# 原始侧表 embedding 的 Slice 审查

原始 Level1 第 4 步在物理流 47 的 `aclnnEmbedding` 组内观察到 1,024 个 Slice 和 1,027 个 GatherV2。反向 `aclnnEmbeddingDenseBackward` 内另有 1,024 个 Slice。设备 Slice 从 int64 [19,4096] 取出 [19,4]，其中 4096 = 1,024 张表 × 每表 4 个特征。传到 op-plugin 的索引是 `feature_ids[:, :, table_index, :]`，同一表的相邻 token 首元素相隔 4,096 个 int64。物理 Slice 在 CANN 的复合算子 executor 内运行，Host 的一个 enqueue/dequeue 覆盖了索引物化和实际查表。

D05 做过安全的描述符基址和剩余 offset 规范化。两步仍是 1,027 个前向请求、2,051 个设备发射，1,024 个 Slice 换成 1,024 个 AsStrided。D02 的反向也保留相应物化核。改写地址与剩余 offset 能复用 executor；将行跨度从 4,096 错报成 4 会读到其他表的 ID。[D05 原始逐流结果](../../dynamic-infra-20261127/analysis/forward-rebase-l1-ab/RESULTS.md) · [D02 原始逐流结果](../../dynamic-infra-20261127/analysis/index-rebase-l1-ab/RESULTS.md)

训练镜像已有 `aclnnEmbedding`、`aclnnEmbeddingDenseBackward`、`aclnnGather`、`aclnnGatherV2`、`aclnnGatherV3` 和 `aclnnGatherNd`。目标设备的实机回执为 Ascend910_9362，镜像随附此型号平台配置 `NpuArch=2201`。以下公开 ops-nn 对应版本为 v9.1.0，镜像中的专有二进制无法借此确认逐行相同。[本机头文件](embedding-headers.txt) · [本机平台配置](embedding-platform.txt) · [目标卡实机 SoC 回执](../../dynamic-infra-20261127/pr42634/stock-fold-probe.json)

| 公开 v9.1.0 函数 | 对实际输入的限制 |
|---|---|
| `aclnnEmbeddingGetWorkspaceSize` | 直接执行 `l0op::Contiguous(indices)` 后再进入 GatherV2。文件定义的无物化 helper 要求 Regbase、两份二维输入；此版本 workspace 函数未调用该 helper。 |
| `aclnnEmbeddingDenseBackwardGetWorkspaceSize` | 在处理 index 类型转换和调用 EmbeddingDenseGradV2 前执行 `l0op::Contiguous(indices)`。 |
| `aclnnGatherV2GetWorkspaceSize`、`aclnnGatherV3GetWorkspaceSize`、`aclnnIndexSelectGetWorkspaceSize`、`aclnnGatherNdGetWorkspaceSize` | 对 index 执行 `l0op::Contiguous`。 |
| `aclnnGatherGetWorkspaceSize` | GatherElements 的一般 strided 分支要求 Regbase。v9.1.0 的 Regbase 集合只列 DAV_3510。 |

实现来源为公开镜像 commit `ceb4536a2bd6fc99b85aec9d0fdcc0f470376292`。[Embedding](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/ceb4536a2bd6fc99b85aec9d0fdcc0f470376292/index/embedding/op_api/aclnn_embedding.cpp#L269) · [Embedding backward](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/ceb4536a2bd6fc99b85aec9d0fdcc0f470376292/index/embedding_dense_grad_v2/op_api/aclnn_embedding_dense_backward.cpp#L346) · [GatherV2](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/ceb4536a2bd6fc99b85aec9d0fdcc0f470376292/index/gather_v2/op_api/aclnn_gather_v2.cpp#L203) · [GatherV3](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/ceb4536a2bd6fc99b85aec9d0fdcc0f470376292/index/gather_v2/op_api/aclnn_gather_v3.cpp#L239) · [IndexSelect](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/ceb4536a2bd6fc99b85aec9d0fdcc0f470376292/index/gather_v2/op_api/aclnn_index_select.cpp#L197) · [GatherNd](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/ceb4536a2bd6fc99b85aec9d0fdcc0f470376292/index/gather_nd/op_api/aclnn_gather_nd.cpp#L200) · [GatherElements](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/ceb4536a2bd6fc99b85aec9d0fdcc0f470376292/index/gather_elements_v2/op_host/op_api/aclnn_gather.cpp#L384) · [Regbase](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/ceb4536a2bd6fc99b85aec9d0fdcc0f470376292/common/inc/op_api/aclnn_util.h#L32)

已查接口中没有证据支持在本机保持现有数据布局、只换 Host 描述符便可省一次提交。另做跨表重排需要读取和写入实际索引，并确保后续每次使用时原 storage 的内容仍相同。已测 D05 收益与这种新的数据物化方案应分别计量。
