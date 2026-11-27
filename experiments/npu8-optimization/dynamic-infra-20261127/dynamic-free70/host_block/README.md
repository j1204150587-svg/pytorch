# HSTUBlock 的 C++ eager 调度

四层 HSTU 每步各执行一次前向。候选在 `HSTUBlock.forward` 接受真实尺寸张量，把两个 RMS、UVQK 投影、注意力、门控和输出投影按原顺序交给 `at::` 操作。每个操作继续进入原有 dispatcher、NPU kernel 和 autograd；C++ 操作以 `CompositeImplicitAutograd` 注册。动态序列长度仍参与 reshape、下三角掩码和除法。若层配置、梯度模式、dropout 或子模块 hook 不匹配，沿原 Python 路径执行。[实现](host_block.cpp) · [守卫](block_dispatch.py) · [训练接入和计数](stage.py)

在单个 BF16 HSTUBlock 上逐位核对输出、输入梯度和六项参数梯度。[1,19]、[2,14]、[2,22]、[4,31] 在逻辑 NPU 8 均通过。本机 CPU 上 [2,14] 的原实现和候选都产生 838 个 ATen 事件，其有序名称逐项相等。设备侧实际物理流和任务序列须另外从整步 trace 读取。[NPU 原始结果、编译命令及哈希](npu.json) · [CPU 结果与 ATen 序列核对](cpu.json)

| 微测顺序 | 原实现，μs/块 | C++ eager，μs/块 | 耗时降幅 |
|---|---:|---:|---:|
| A/B | 2739.67 | 2096.55 | 23.47% |
| B/A | 2450.78 | 2093.45 | 14.58% |
| A/B | 2429.02 | 2142.20 | 11.81% |
| B/A | 2441.67 | 2123.44 | 13.03% |

每个同步窗口 32 次完整块前反向，轮流取上述四个真实形状。端到端及 Free 的结论以同配置完整训练的双序结果为准。[微测方法](micro.py)
