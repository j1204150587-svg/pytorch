# 整段 eager BF16 RMSNorm

本候选适用连续的 `[B,L,64]` 与 64 元素权重。一个 Triton program 处理一个真实 token。前向在平方、均值、加 epsilon、rsqrt 和两次乘法后各保留原 BF16 边界；反向在 dX 的每条原求导边界显式转 BF16，保存原生 `sum_to_size` 跨 token 汇总 dW。采用的 `rsqrt` 和规约实现必须通过独立位级门。CPU 的手写导数对照在四个形状和四类输入的 16 例全部通过。[CPU 逐例](cpu-micro.json)

Direct B 实机 Level0 #4–6 中，九处 RMSNorm 各匹配 6 个连续顶层前向标记、6 个由 autograd 序列号配对的反向节点。前向 54 次设备提交包括 pow、mean、adds、rsqrt 各 9 次和 mul 18 次。54 个反向 evaluate 节点内每步真实提交 139 次，含 9 次 SDMA 和 130 次计算；其中部分 add 属于节点输出的梯度累加。三步中前向标记和官方 Free 的交集为 1.445、1.410、1.471 ms，反向为 3.763、3.687、3.758 ms。两者合计约 5.18 ms/步，表示观测到的交集。设备任务通过同线程源时间戳处于目标 Host 标记的 `async_npu` flow、物理流及设备起点唯一匹配。[按步、核和流的计数](direct-stock-counts.json) · [复算](scope_counts.py)

同序 [1,19]、[2,14]、[2,22]、[4,31]，含随机单位权重、随机权重、近 epsilon 和零输入。输出、dX、dW 逐个 BF16 位型比较，首例不同时会保留完整四形状诊断。正常 kernel 未启用中间值写回；只有失配才另跑 debug kernel，对照 6 个前向阶段、12 个反向阶段，反向还分别使用原生及候选的前向中间值。全数通过才计时，四轮 AB/BA、每轮每条腿 96 次。完整训练仍需另验 loss、所有梯度、更权、溢出跳步和实际设备提交次数。

```bash
python micro_gate.py --device npu --out npu-micro.json --repeats 96
```
