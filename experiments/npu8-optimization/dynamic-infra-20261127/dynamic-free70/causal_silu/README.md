# 动态 eager HSTU 的 causal SiLU attention

基线在四个 HSTUBlock 内分别执行 QKᵀ、乘 0.25、SiLU、因果 mask、除以当步序列长度 L、与 V 相乘。因果区间包括对角线。Q/K/V 为同一 `[B,L,256]` 真实 projection 的三个切片，形状 `[B,4,L,16]`、offset 分别为 128、192、64。微测从 BF16 的 `F.linear` 生成 projection，由 split/view/transpose 生成相同布局。上游梯度由 `[B,L,4,16]` 转置产生。

全融合方案面向 D=16、L=12–32，每个 batch/head 使用一个 32 行、16 维的程序块。只在寄存器内 tile 尾部补零，所有张量均按实际 B/L 分配。前向一次 Triton 调用保存舍入后的 scaled scores 和归一化 scores。反向一次 Triton 调用计算 dQ/dK/dV。每次矩阵乘法、乘常数、SiLU、除以 L 和 SiLUBackward 的原 BF16 边界均执行显式舍入。两种后端的数学函数、矩阵乘累加顺序和非有限值编码必须用逐位门进一步验证。

部分融合方案保留原 ATen matmul、标量缩放、SiLU 和相应反向，合并生成布尔三角掩码、反相、masked_fill 和除以实际 L。前向和反向各调用一个向量内核，反向把因果区域以外的值直接置为正零。非标准形状或布局走原 eager 实现。

CPU 模拟在 60 个实际 shape 及 6 个附加 signed-zero 或非有限样例上逐 BF16 位对照了 Y、dQ、dK、dV。部分融合 66/66 一致。全融合模拟 61/66 一致，其中两个有限样例各有一个 dQ 值不同。[CPU 部分融合](cpu-partial.json) · [CPU 全融合](cpu-full.json)

每个候选独立运行，结果文件每个用例更新一次。成功通过全部用例后才进行四个交替 A/B、B/A 无 profiler 窗口，每个窗口只在首尾同步，包含实际前向和三组梯度。用完整 schedule 重查首次编译后所有 shape。各次测试记录原始 stride、storage identity、NPU format 和源码哈希。

```sh
R=/artifacts/dynamic-infra-20261127/dynamic-free70
for mode in partial full; do
    python -B "$R/causal_silu/micro.py" --device npu --mode "$mode" --repeats 48 --out "$R/attention-$mode.json"
done
```

正式配方 `runner_tools/recipes/attention-mask-div.json` 以 01 动态 eager 为基线，两腿相差 `NPU_DYNAMIC_CAUSAL_ATTENTION`。每次迭代应在四层各触发一次，`causal_attention_counts.json` 记录逐层调用、命中和全模型回退。Level1 查找 `HSTUAttentionOriginal`、`HSTUAttentionMaskDiv`、`MaskDivBackward`、`mask_div`，按相同步骤和 stream 核实实际 device task 数与关键空档。
