# 单 rank INT32 MAX 的 HCCL 提交

同一 V2 版 `libtorch_npu.so` 中切换 `NPU_HCCL_RANK1_SCALAR`。基线启用原始 01 分块 overflow 和 Task Queue Mode 1。正反序的单腿训练都在预热 10 步后用首尾同步窗口测量 50 步，每腿 136 个样本、2,994 个 token。八步正确性门逐步检查 1,052 个梯度，三次 overflow 跳步、五次权重更新，最大 loss 绝对差 0.000003815、norm 差 0。正序 Level1 候选 8 步命中 8 次、健康检查回退 0 次。[正确性门](gate-comparison.json) · [激活计数](native-l1-ab/02-B/hccl_rank1.json)

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 634.395 ms/步 | 604.263 ms/步 | +4.986% | 591.893 ms/步 | 92.038% | 580.622 ms/步 | 91.723% | +0.343% |

AB 的吞吐提升 1.433%，BA 提升 8.787%。合并窗口的样本/s 为 4.288→4.501，token/s 为 94.389→99.096。第 4–6 步实际形状为 [1,19]、[2,14]、[2,22]。两序合并后 Stage 为 643.093→633.019 ms/步，Computing 为 51.073→52.318 ms/步。Free 占比的相对变化分别为 AB 改善 1.268%、BA 回退 0.616%。[原始窗口和计算](recomputed.json) · [复算程序](../../ipg_range_copy/paired_metrics.py)

第 4、5 步中每步物理 stream 45 各少两笔 `MEMCPY_ASYNC / SDMA_SQE`。A 的 INT32 MAX 和 FP32 SUM 各对应两笔，B 仅保留 FP32 两笔。流 44、47 全部设备任务的名称、类型和顺序逐项相同。A 的 INT32 HCCL Host 执行分别为 147.01、135.79 µs，与官方 Free 重合 117.10、125.05 µs。B 同一 consumer 保留 `HcclAllreduce` 队列项，服务从 162.85→16.87 和 152.31→23.21 µs。FP32 的排队在这些步反而变长。Acl/HCCL 本地实测解释了约 0.14 ms 的 consumer 缩短与每步两笔设备控制任务减少；完整窗口相差约 30 ms/步，其余变化尚未定位。[分步实际设备序列及队列交集](native-l1-ab/physical.json) · [直接复算](../../ipg_native_views/paired_physical.py)
