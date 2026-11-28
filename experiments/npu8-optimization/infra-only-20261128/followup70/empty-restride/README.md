# PTA 空张量工厂的连续 stride 计算

`NPUNativeFunctions::empty` 和 `empty_with_format` 先调用 `TensorImpl::set_sizes_contiguous`，该函数已经计算连续 stride。补丁为已执行这一步的路径省去后续重复计算，保留单维零长等特例。正反两序都开启 01、双向 embedding 描述符、65,536 项 CANN 缓存和 Task Queue Mode 1，只切换 `TORCH_NPU_EMPTY_SINGLE_RESTRIDE`。[实测补丁](empty-restride.patch) · [构造器微门、代码指纹及训练计数](repro.tgz)

54 个工厂、shape 与 dtype 组合通过两腿比对。原始八步审计中，A 保留 63,017 次计算。B 跳过 53,185 次，保留 9,832 次。梯度、norm、最终 BF16/FP32 权重和 AdamW 相同，最大 loss 绝对差 7.63×10⁻⁶。该训练序列为 8 次有限值更新。

实编命令从同镜像的 `/artifacts/infra-only-20261128/followup70-20261128` 执行，使用仓库 blob 校验原始源文件。

```bash
python3 -B release-ring/build.py empty-restride/empty-restride.patch torch_npu/csrc/aten/common/TensorFactories.cpp 1a327c97e16098e5e334bf8c603ec4eb70fe21f4
```

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|376.789|379.648|−0.76%|459.544|89.453%|421.878|88.788%|+0.74%|
|388.577|393.736|−1.33%|468.173|89.741%|430.825|88.846%|+1.00%|

表中时间为 ms/步。两行按正序 A/B、反序 B/A 排列，均以 A→B 展示。无 profiler 同步 50 步分别处理 136 个样本、2,994 个 token。吞吐为 7.219→7.165、7.000→6.908 样本/s，以及 158.922→157.725、154.101→152.081 token/s。配对节省 −2.860、−5.159 ms/步，均值 −4.009、样本标准差 1.626 ms/步。Level0 的三步形状为 [1,19]、[2,14]、[2,22]，Stage 为 513.724→475.153、521.694→484.910 ms/步，Computing 为 54.060→53.173、53.415→53.957 ms/步。[完整双序原值](metrics.json) · [运行命令、CSV 与窗口](windows-and-commands.tgz)

Level1 的第 4、5 步分别有 7,743、7,744 个 `empty_tensor`。两步工厂 p95 为 5.61→5.51、5.55→5.50 µs，工厂区间并集与全流空档的交集为 21.623→21.287、22.593→22.170 ms。相同物理流的设备工作、控制序列和 ACL 次数均一致，kernel launch p95 为 7.71→7.46、7.80→7.73 µs。第 4 步 host 入队数为 9,931→9,928，第 5 步一致。早窗 Free 下降，但两序稳态均回退，归入「尝试无优化路径」。十二腿卡前后映射和芯片进程均通过核对。[A 工厂分位数及交集](factory-01-A.json) · [B 工厂分位数及交集](factory-02-B.json) · [逐流与提交](physical-comparison.json) · [原始数据目录及 SHA-256](evidence-index.json)
