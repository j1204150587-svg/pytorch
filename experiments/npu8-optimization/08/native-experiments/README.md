# 08 PTA 原生路径实验

在 TorchNPU `38e483ee79ec8a3286517808740a811c0f0ae66c` 上应用 [候选补丁](queued-copy-replay.patch)。对照使用同一构建，固定 08 的 HSTU 前反向图和两张统计图。实验入口分别控制两个开关。

| 开关 | 生效路径 |
|---|---|
| `NPU_EXPERIMENT_QUEUE_COPY_REPLAY=1` | 08 helper 调用 `NPUGraph._try_replay_with_copy()`，在 Task Queue Mode 1 共享队列中合并一次 D2D 和图回放。调用保持同一流，缓存分配器记录源和目标的流依赖。图销毁或改写前等待回放流上的设备工作完成。 |
| `NPU_EXPERIMENT_FLAT_ND=1` | 仅在多个连续 ND 分片按序相邻且共用存储时，用一个连续视图的 clone 替代逐分片准备后的 Cat。输出仍有独立存储。其他布局沿用原路径。 |

轮询、绑核、训练模型、DeepSpeed 调用、梯度布局及数值表达式沿用 08。两开关缺省关闭，基线和不同候选由独立进程读取开关。

## 复算

在带原 08 run 目录和历史语义 verifier 的宿主机 artifacts 下运行 [准备入口](prepare.py)，可复用同一容器及逻辑卡 8 锁。原 wheel 构建打开 C++11 ABI，关闭 TorchAir 和 RPC，使用 Ninja。候选 wheel 安装在隔离目录。运行 [微型正确性测试](test_pta.py) 与 [八步训练对照](compare_gate.py) 后，运行 [串行配对](run_suite.py) 中的 `steady` 和 `level0` 阶段。

稳态计时预热 10 步、同步计时 50 步。Level0 取第 4–6 步，按同窗 Stage 复算 Free 占比。用 [证据打包](pack_evidence.py) 保存分组 JSON、原始稳态 metrics、Level0 CSV 及卡况，用 [复算入口](summarize.py) 重新计算各项收益。
