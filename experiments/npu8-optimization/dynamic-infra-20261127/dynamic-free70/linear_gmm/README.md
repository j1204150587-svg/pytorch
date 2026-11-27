# 动态线性层反向的两次矩阵乘

实测配置的四个 HSTU 块各含 `64→256` 和 `192→64` 两个有 bias 的线性层。stock 的第 4、5 步分别是 `[1,19]` 和 `[2,14]`。两步均有 8 个 `AddmmBackward0`，每个含两笔 `aten::mm → Enqueue@aclnnMm → aclrtLaunchKernelWithHostArgs → MatMulV2`。两笔均落在物理流 47 的相邻工作槽位，两个实际形状的 8 对槽位一致。8 个反向 Host 范围合计 1.474/1.445 ms，后继为这 16 个 MM 的全流空档合计 1.714/1.610 ms。空档还包含其他 Host 工作、队列和设备依赖。[实测调用及流内编号](stock-pairs.json) · [复算](stock_pairs.py)

设实际长度 `M=B*L`。原反向需要 `dx=dy@W` 和 `dw=dy.T@x`。实验用 op-plugin 的多输入、多权重、多输出 `npu_grouped_matmul` 一次表达这两笔乘法，两张左操作数分别是 `dy` 和 `dy.T.contiguous()`，两张右操作数为 `W` 和 `x`。保持原 BF16 数据与实际矩阵大小，`group_list=None, split_item=0, group_type=-1`。调度产品路径取决于本机 CANN 导出版本。API 请求一次与实际物理 kernel 数分别核实。[op-plugin 实现](https://github.com/Ascend/op-plugin/blob/eaf6b55812a1998a4a40e485e3b2861a40e9c2f6/op_plugin/ops/opapi/GroupedMatmulKernelNpuOpApi.cpp) · [分组和布局条件](https://github.com/Ascend/op-plugin/blob/eaf6b55812a1998a4a40e485e3b2861a40e9c2f6/docs/zh/custom_APIs/torch_npu/torch_npu-npu_grouped_matmul.md)

微测先取同一个实际输入 `F.linear(x,W,bias)` 的 autograd 结果，分别核对两个普通 MM 和候选 GMM 的 dInput、dWeight。输出逐个 BF16 字比较，另报最大绝对误差。4 个实测形状 `[2,16] [1,19] [2,14] [2,22]` 覆盖 M=32/19/28/44，每个形状运行两种投影。CPU 的八组普通 MM 两份梯度均与 autograd 逐字相同。NPU A 的计时只含对应两次普通 MM，B 的计时含动态转置复制、分组参数、实际设备提交；每轮同步窗口首尾，预热后 AB/BA 各两次。另记录首次调用并同步的耗时、转置复制字节、实际 stride，以及 M=32/19 的独立 Level1。每种形状、投影使用独立进程，保留失败回执。若代表实测形状、两份梯度、布局、物理提交或者含复制净步时不足以支持替换，停止训练层接入。完整 shape 序列可用 `--all`。

```sh
TASK_QUEUE_ENABLE=1 python3 "$R/linear_gmm/micro.py" --device npu \
    --schedule "$W/configs/dynamic_shapes_npu8.json" --output "$R/linear-gmm-screen-v1" \
    --repeat 32 --profile-case
```
