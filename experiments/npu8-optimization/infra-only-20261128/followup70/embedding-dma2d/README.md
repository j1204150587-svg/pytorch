# 用 CANN 二维 D2D 拷贝连续化侧表索引

## 候选与原训练锚点

单个侧表索引来自 INT64 `feature_ids[:, :, t, :]`。四个相邻 ID 共 32 字节，连续两行在源张量中相隔 32,768 字节；目标行跨度 32 字节，高度为当前 `B×L`。候选仅对当前一个表的一次访问考虑 `aclrtMemcpy2dAsync`，源起点加 `32×t`，目的缓冲独立且容量为 `32×B×L`。

在双向视图重定位的原动态 eager Level1 中，第 4 步 `[1,19]` 和第 5 步 `[2,14]` 都在流 47 上出现 1,024 笔前向 `aclnnEmbedding_AsStridedAiCore_AsStrided` 与 1,024 笔反向 `aclnnEmbeddingDenseBackward_AsStridedAiCore_AsStrided`。各请求使用一个设备工作项从 strided 索引取出连续数据。实验希望以较低 Host 提交成本完成同一工作；若设备拒绝、数据不一致，或每次转换需要更多物理提交而未见更短路径，则在微门结束。

| 步骤 | 方向 | 设备执行合计 ms | 相邻前置全流空档合计 ms | 空档 p99 µs | 名义 ACL 已返回 ms |
|---|---|---:|---:|---:|---:|
| 4 · [1,19] | 前向 | 2.462 | 46.037 | 69.06 | 7.169 |
| 4 · [1,19] | 反向 | 2.830 | 3.852 | 19.16 | 3.186 |
| 5 · [2,14] | 前向 | 2.471 | 42.779 | 79.10 | 20.287 |
| 5 · [2,14] | 反向 | 2.772 | 4.426 | 21.06 | 4.388 |

相邻全流空档由设备工作区间并集计算。用 step、物理流、任务号及名称匹配原始 CSV 后分解 producer、入队和等待、consumer、ACL 调用以及返回后阶段。上述空档量并非候选预期收益。[原训练分位数和各阶段](stock.json) · [复算脚本](audit.py)

## 同版源码

CANN 9.1.0 实包 `acl_rt.h:2383` 导出 `aclrtMemcpy2dAsync`。文档注释只写 host/device；运行时公开源码的 `CheckMemcpy2dParam` 明确将 `ACL_MEMCPY_DEVICE_TO_DEVICE` 映射为同名 RT kind，`ApiErrorDecorator::MemCopy2DAsync` 再检查内存位置和方向。后端 `ApiImpl::MemCopy2DAsync` 以 `width×height` 为剩余字节，每次调用 `Memcpy2DAsync` 后将源、目的偏移各推进一个 pitch 并扣除 `realSize`。STARS 的每次调用分配一项 MEMCPY task 并提交。在同设备 SDMA 分支 `MemcpyAsyncTaskInitV2` 明确赋 `size=width`，所以这条实现会逐行下发。[前端与实包指纹](installed.txt) · [同版源码、行号与哈希](source.txt) · [v9.1.0 任务构造](https://github.com/hicann/runtime/blob/0b4af8a31c75a1f1f43638a8f75b9e3c6182e031/src/runtime/core/src/task/task_info/memory/memory_task.cc#L531)


## 实机结果与停止条件

板 4、芯片 0、逻辑卡 8、容器可见设备 0。用同一训练容器与两层共享锁，轮换两个流和表号 0、15、1023，覆盖 60 个合成实际 shape。在每个 shape 先验证独立 PyTorch 连续化，再用普通 32 字节 D2D 作为正对照，最后执行二维拷贝。均使用独立、已分配、逐字节核对边界的设备缓冲。二维调用 60/60 返回 0，目标流同步 60/60 返回 0，全部输出与 CPU 原切片逐位一致。[微门源码](micro.py) · [按 leg 的原始结果与 trace 核验](result.json)

| 实际 shape | 表号与偏移 B | 物理流 | 独立连续化 / 单行正对照 / 二维 DMA 设备任务数 | 独立连续化 / 二维 DMA 的各项设备执行时间之和 µs |
|---|---:|---:|---:|---:|
| [1,19] | 0 / 0 | 47 | 1 / 1 / 19 | 3.38 / 13.40 |
| [2,14] | 15 / 480 | 45 | 1 / 1 / 28 | 3.46 / 21.30 |
| [2,22] | 1023 / 32736 | 47 | 1 / 1 / 44 | 3.18 / 35.18 |

每组 Level1 的 CANN 轨都有一笔 `aclrtMemcpy2dAsync`。实际 `Ascend Hardware` 轨录到连续 task ID 1734–1752、1942–1969、1758–1801，每项 `MEMCPY_ASYNC`。独立 PyTorch 正对照对应 `aclnnInplaceCopy_SliceAiCore_Slice`；原训练目标则是 embedding executor 内的一笔 AsStrided。两种连续化均只提交一个设备任务。60 组无 profiler 微门的二维调用 Host p50/p99 为 66.443/122.835 µs，随后目标流同步 p50/p99 为 77.593/310.332 µs。微门各腿固定顺序、逐次同步，独立 PyTorch 腿包含 Python 视图和首次执行开销，故未以跨腿 Host 时间推断 HSTU 提速或回退。

**尝试无优化路径。** 在这台 910_9362 上，二维 D2D 的数值与偏移正确，但 Runtime 的每行一个设备任务放大了真实提交数，单次调用的三个实测 shape 设备工作之和也都更高。若逐个替换原前反向各 1,024 次连续化，按实测行数推算，第 4、5 步分别新增 36,864、55,296 个设备任务；该数字是投影。候选止于算子门，未修改原训练、未做真实训练激活或 E2E/Level0，未发布分支。

[原始命令、shape、完整微门结果、Level1 trace 与设备收据](probe-evidence.tgz)，SHA-256 `5672b99c928ce5a92d1e27960c562a5f14eb56eca1806e1b67498c0d14be2dca`。运行 `python audit.py probe probe-evidence.tgz --out result.json` 可从原始轨道重新计数。在原训练 B 腿的 `kernel_details.csv` 与 `producer-coverage/physical-gap-coverage.csv.gz` 上执行 `python audit.py training <kernel CSV> <完整空档 CSV> --out stock.json` 可复算候选锚点。

