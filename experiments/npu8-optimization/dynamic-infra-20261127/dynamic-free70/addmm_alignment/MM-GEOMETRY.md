# HSTU 反向投影的 Mm 描述符

在校时 Level1 第 4、5 步，`aclnnMmGetWorkspaceSize` 与后续 Matmul 前的全流空档分别重合 0.604–0.620、0.552–0.576 ms。两步的实际形状为 [1,19]、[2,14]。这一统计含反向的不同 Mm 请求。第 4 步流 47 的 task 4865 对应队列项 10636，单次 workspace 与空档重合 85.37 μs。[逐任务的时间边界](../diagnostics28/clock-analysis/bounded_gaps.csv.gz) · [按算子求和](../diagnostics28/clock-analysis/common_shift_bounds.csv)

候选定位到 `op_api::mm` 中 `EXEC_NPU_CMD(aclnnMm, ...)` 的功能式 ND 分支，以及 `add_param_to_buf` 和 `ConvertType` 的只读权重处理。每步四层八次的输入梯度投影读取连续的 BF16 参数矩阵，形状为 [256,64] 或 [64,192]。同层的权重梯度投影使用不同的两个输入。实机 ZeRO 参数布局用于重建 CPU eager 反向，记录了全部 960 次 Mm 的输入和输出几何。[逐次几何](mm-results.json) · [CPU 复算](mm_geometry.py)

将这八个真实参数视图的基址设为 128 字节对齐的实际数据地址、偏移归零并按有效元素数限定范围，480 次只读权重请求的几何由 416 种降至 104 种。十步预热后的 400 次只读权重请求中，已经出现过相同几何的次数由 64 次增至 316 次。实际 executor 命中、延迟以及设备空档须在目标 wheel 单独测量。

在另一台 Windows CPU 的 PyTorch 2.9.1 上重跑相同的 60 组前反向，所得 960 条记录与目标镜像的 JSON 原始字节完全相同。该环境用已核对形状、dtype 和 layout 的具体上游梯度直接调用 autograd engine，仍由相同的 dispatcher hook 记录每笔 `aten.mm`。

NPU 探针使用两套独立参数和两条 stream，按 CPU 记录的形状、stride 和 offset 构造原视图与相同指针的两种描述符。逐位比较 480 个输入梯度结果，四个新进程做 A/B、B/A，再用同一物理流比较先见及复访的 workspace 次数和设备工作。一旦有 BF16 差异，或两个顺序的同步窗口与 workspace 未见可复现改善，该提案停止在微测。[探针及全部条件](mm_rebase_probe.py)
