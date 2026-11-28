# 动态 eager 侧表 embedding 反向使用 BF16 归约内核

## 原路径和候选

在相同的双向侧表视图重定位前置项上，原版 `aclnnEmbeddingDenseBackward` 会将 BF16 梯度转成 FP32，执行 `EmbeddingDenseGradV2`，再将输出转回 BF16。在原训练 Level1 中，第 4 步 `[1,19]` 和第 5 步 `[2,14]` 各有 1,024 次侧表归约，物理流都是 47，`Block Num=40`，输入类型均为 FLOAT、INT32、INT32。另有三个来自其他表的反向。合计每步 1,027 个反向请求包含 3,081 个 Cast。

私有 Host 入口对安装包已有的小维度 `EmbeddingDenseGradV2` 传入 BF16 梯度。该内核在片上转为 FP32、以 FP32 归约、最后按 `CAST_RINT` 写回 BF16。索引连续化和 INT64 到 INT32 转换仍由原实现处理。选择条件包含 DAV_2201、Ascend910_9362、关闭确定性、行数 64、维数 64、每表 4 个 ID、明确的 padding 和 scale 条件。构建使用 PTA `38e483ee79ec8a3286517808740a811c0f0ae66c`、op-plugin `eaf6b55812a1998a4a40e485e3b2861a40e9c2f6`。私有 Host SHA-256 为 `eab824cad8f837a24c090a33e1be56e25caa9f8c19d936753db979af7106673e`，连接好实验入口的 libtorch_npu SHA-256 为 `53d28d5b34ab209365f0e98210235304a4e041d25ee05026ea90d5cf1b36b051`。[设计与源码链接](设计.md) · [Host 构建收据](build-receipt.json) · [PTA 构建收据](plugin-receipt.json) · [PTA 补丁](route.patch)

## 原机首轮

在板 4、芯片 0、逻辑卡 8、容器可见设备 0 下，每组保持同一份梯度和 strided 侧表索引，依次执行 A–B–B–A，共四次。对 60 种实际尺寸分别使用精确小整数和随机 BF16 梯度，轮换两条流及 padding −1/7。120 组均有 4,096 个 BF16 输出逐位一致。

在第 121 组 `[2,22]` 高动态范围强抵消输入中，原腿两次相差 28 个元素，候选两次也相差 28 个，跨腿两组分别相差 19/37 个。位置 `[0,2]` 依次输出 5、9、3、5。两腿 workspace 分别为 16,841,216 和 16,795,136 字节。脚本到这里停止，最初脚本末尾的两个独立输入组没有执行。[首轮原始结果](first-results.json) · [生成与运行代码](micro.py)

小维度内核的 `SortIndices` 按核排序索引，`AtomicAddInUb` 在本地对同一 ID 的行顺序累加，`CopyOut` 通过 `SetAtomicAdd<float>` 将部分和累加到全局输出。FP32 腿写 FP32 输出，BF16 腿写 FP32 workspace 后再转存，并额外经过初始化及屏障。对同一个 ID，由哪些核按怎样的到达顺序提交部分和，影响强抵消的结果。数值观察和源码不能提供单次执行中各核原子写的完整先后关系。[tiling](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/ceb4536a2bd6fc99b85aec9d0fdcc0f470376292/index/embedding_dense_grad_v2/op_host/embedding_dense_grad_v2_tiling.cpp) · [小维度内核](https://github.com/Atlas-Ascend-Fork/ops-nn/blob/ceb4536a2bd6fc99b85aec9d0fdcc0f470376292/index/embedding_dense_grad_v2/op_kernel/embedding_dense_grad_v2_small_dim.h)


## 受控数值门

固定实际 shape `[2,22]`，用 strided 侧表索引、padding −1、两条流轮换，对每组保留 4,096 个输出并依次执行 A–B–B–A。22 组的 A–A、B–B 和两次跨腿比较全部逐位相同。

- 每个输出坐标最多有一个非零输入，16 组覆盖全部 65,536 种 BF16 位编码，包括子正规、正负零、无穷及 NaN 的不同位型。比较对象是两条路径的最终输出位型。
- 两项相加覆盖正负半 ULP 舍入、抵消、上下溢和混合无穷。另以 5 种二次幂缩放验证可精确表示的小整数高重复加和。

这排除了受控用例中单值转换及上述二项舍入差异；多核写回顺序仍需单独观察。[执行代码](controlled.py) · [22 组原始结果](controlled-results.json)


## 训练 trace 中两个 Cast 的位置

以双向侧表视图重定位的 Level1 B 腿为入口，用同一步、流、任务号、任务名，把物理 `kernel_details.csv` 的 dtype/shape 与所有设备流的工作区间并集配对。表内每个任务类别每步有 1,024 笔。阶段分解采用名义校时；空档的成因及可消除时长仍要依赖候选实测。

| 步和 shape | Cast | 设备工作合计 ms | 前邻全流空档 ms | 空档 p50 / p99 / 最大 µs | 空档内 ACL 已返回 ms |
|---|---|---:|---:|---:|---:|
| 4 · [1,19] | grad BF16 → FP32 | 1.408 | 25.366 | 22.36 / 94.56 / 156.28 | 10.679 |
| 4 · [1,19] | 输出 FP32 → BF16 | 1.403 | 1.231 | 1.00 / 7.84 / 14.98 | 1.221 |
| 5 · [2,14] | grad BF16 → FP32 | 1.409 | 25.330 | 23.44 / 93.00 / 169.94 | 22.046 |
| 5 · [2,14] | 输出 FP32 → BF16 | 1.479 | 1.501 | 1.02 / 9.96 / 236.56 | 1.495 |

两个步骤的输入 Cast 前仍有 producer 正在准备、入队、等待 consumer，以及 consumer 从出队到发起和完成 ACL 的时间；分项 ms 和原 CSV 校验值可复算。一个 ACLNN 请求提交多项设备工作。[物理任务清单](hstu-stock-task-counts.json) · [分阶和延迟分位数](stock-gap-results.json) · [复算脚本](stock_gaps.py) · [原始逐空档记录](../../producer-coverage/physical-gap-coverage.csv.gz)


## 固定地址与归约顺序

原始 `[2,22]` 强抵消样本按相同 RNG 重新生成。在每一路 stream 内固定 grad、索引、输出和 workspace 地址，按 `ABBABAAB` 循环 6 次，每腿 24 次。第一条流 A/B 分别得到 23/23 个不同的整张量 SHA-256，另一流得到 22/24 个。两条流上两腿均在 37 个输出坐标出现自身波动。按每个坐标比较两腿的观测取值集合，均有交集，未观察到两腿各自稳定而互异的坐标。集合相交不能推导相同的结果分布。

另将同样的三个有限贡献 `2^24、1、−2^24` 放在同一个或不同的物理 block，保留训练中同一 token 的四个特征共享梯度这一结构。对 `[2,22]`、`[1,19]`、`[4,21]` 分别测试两个流，每组每腿 16 次。放在一个 block 时，两腿目标位置 192 次计算全部得到 0。分散到不同 block 后，原版在 6 组均出现 0 和 1；候选在 `[2,22]` 的两个流也出现两种结果，在另两个 shape 的四组均 16/16 为 1。结合源码和同核对照，归约分组可以解释同腿波动。设备 trace 能识别 40 个 block，未记录每一笔原子加法的到达次序。[固定地址与输入重建](repeat_order.py) · [逐坐标结果与直方图](replication-summary.json)

局部 Level1 的 12 个 A–B–B–A 标记全部匹配物理轨道。每次候选内部实际任务从 6 减至 4，Cast 从 3 减至 1，仅保留索引 INT64 → INT32。每个标记中另有共同的 1 笔预填充输出，计入则为 7→5。相同步位置两腿始终同一流，`[2,22]` 和 `[4,21]` 为物理流 47，`[1,19]` 为流 45，归约均为 40 blocks。这里测得的是 CANN Host 入口的设备提交数。

## 尝试无优化路径

仅靠当前 shape、stride、dtype、padding 无法在允许的任意索引值和梯度下证明逐位等价。以“每个输出 ID 至多两个贡献”这一充分条件考察实际训练输入，按原 rank 0 的 CPU RNG 和脚本顺序重建 60 步、61,440 张次索引，只有 65 张次满足（0.106%），有张次的单个 ID 最高出现 23 次。65 是这一特定条件的命中数；张量描述符本身不带 ID 重数、每核 ID 分布或梯度指数范围。发布 Host 路径不能把这个合成种子及表号当作业务条件。

该候选停止在局部语义门，真实训练激活、回退、loss/梯度、overflow、权重更新、无 profiler 端到端和 Level0 Free 均未测。局部试验随动态台账归入「尝试无优化路径」。原始包包含脚本、命令、设备前后检查、完整 BF16 输出、Level1 trace/CSV、shape 序列和训练输入生成代码，SHA-256 为 `0c3182512a750802ab45d44d25f8fb8b339128872953feabb200ab3e5f97471d`。用 `python audit_results.py probe-evidence.tgz replication-summary.json` 可重新验证每次输出哈希、获取逐坐标波动和物理轨道。[原始包](probe-evidence.tgz) · [复算脚本](audit_results.py)

