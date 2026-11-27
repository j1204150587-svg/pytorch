# Addmm 偏置与转置权重的描述符对齐

在 Direct-B 单 rank ZeRO-2 的实机 Level1 中，第 4、5 步的八个 `aten::addmm` 各经过 producer 的 `aclnnAddmmGetWorkspaceSize`。每步流 47 实际提交八个 Cast 和八个 MatMul。第一步的 16 笔任务前设备全流空档合计 1.21170 ms，其中 0.54336 ms 与各自生产者的 workspace 调用重合。第二步分别为 1.26834、0.56923 ms。交集按相同的物理流、task ID、queue ID 连接到原 trace，按时间戳逐笔求交。约 45% 的这组前导空档确有 workspace 调用正在运行。[逐步结果](result.json) · [原始 trace](../../pr42634/raw-traces/direct-publish-v1-l1-ab/02-B/trace_view.json.gz)

op-plugin 的 `addmm` 在本路径将偏置 self、输入矩阵、转置权重和输出一并传给 `aclnnAddmm`。`add_param_to_buf` 将各张量的形状、stride、storage offset、底层 storage 长度及格式计入哈希；`ConvertType` 用同一几何构造 ACL 描述符。已有 512 字节规范化策略只将基址前移 512 的倍数，同时保留真实余数。四层中相同投影的权重 stride 与 bias 长度相同，各自仍有四种不同的余数。[Addmm 上游源码](https://github.com/Ascend/op-plugin/blob/eaf6b55812a1998a4a40e485e3b2861a40e9c2f6/op_plugin/ops/opapi/AddmmKernelNpuOpApi.cpp) · [op-plugin 键与描述符](../../rebase/reference/op_plugin/utils/op_api_common.cpp) · [已实测的对齐策略](../../rebase/index_scope.inc)

| HSTU 层 | UVQK 权重 / 偏置元素偏移 | 两者字节余数 mod 512 | 输出权重 / 偏置元素偏移 | 两者字节余数 mod 512 |
|---:|---:|---:|---:|---:|
| 0 | 4,264,064 / 4,280,448 | 256 | 4,280,768 / 4,293,056 | 384 |
| 1 | 4,293,184 / 4,309,568 | 128 | 4,309,888 / 4,322,176 | 256 |
| 2 | 4,322,304 / 4,338,688 | 0 | 4,339,008 / 4,351,296 | 128 |
| 3 | 4,351,424 / 4,367,808 | 384 | 4,368,128 / 4,380,416 | 0 |

上表使用实机归档的原模型、参数顺序和 torch 的 `_unflatten_dense_tensors` 在 CPU 重建。转置权重保留 storage offset。实机的 1,052 份参数、4,380,544 个元素、侧表首元素偏移 69,696 和连续的 1,024 张表用于交叉校验。目标容器中的 `stage_1_and_2.py` SHA-256 为 `478209eee30499ed06aaa199f8aabe760cc176853e251cf929a0ce1333d68b9f`。第 361、399、756 行分别按 optimizer 参数序收集、选择轮询顺序、按 unflatten 结果重绑 view，单 rank 轮询也保序。每块包含 29,120 个 BF16 元素，下一层起点推进 384 字节 mod 512。[实际训练参数与模型](../working/evidence/direct-publish-v1-l1-ab/02-B/rank000/model_summary.json) · [实机侧表连续性检查](../working/evidence/direct-publish-v1-l1-ab/02-B/rank000/side_slab_counts.json) · [复算脚本](geometry.py)

60 个不同 [B,L] 在 Addmm 处有 52 个不同的行数 B×L。480 个调用在原始权重和偏置偏移下有 416 种投影几何，保留 512 字节余数的规范化后仍为 416 种。64 个可能复用的请求均来自跨步重复的行数，原几何已允许这些命中。仅按这两份参数视图统一描述符所能新增的缓存命中为零，因此没有为这项改动启动训练。跨步实际缓存命中还取决于完整参数、被访问的线程和缓存容量。
