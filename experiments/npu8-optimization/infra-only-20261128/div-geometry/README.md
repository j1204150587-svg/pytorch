# 原始标量 Divs 的 producer 参数准备

## 候选和命中范围

两腿共用 65,536 项 CANN executor 缓存，开启侧表前向和反向 embedding 描述符规范化。实测原始均值反向每步有 1,024 个 `op_api::div(Tensor, Scalar)` 的目标调用；上一个紧凑 Div 在 60 步命中 61,440 次 scalar overload，Tensor overload 命中 0 次。此次固定原来的 `aclnnDivs` 及“BroadcastTo + RealDiv”两项设备工作，只修改命中 CANN 缓存前的参数准备。[原生训练计数](../evidence/raw/20261128T032844Z-ab-no-profiler-gpc183sj/02-B/run.log)

原 op-plugin `hit_cache` 每次序列化算子、输入/输出 shape、stride、类型、storage extent、offset、底层格式、除数、设备、stream 与确定性设置，然后对完整字节计算 hash。候选位于 `DivKernelNpuOpApi.cpp` 的真实 scalar overload。在 Mode 1、普通 BF16 侧表 rank-4 布局和数值 4 下，线程单槽记录原函数实际算出的 hash。每次调用重新核对完整的变化字段并绑定当前两个 storage 地址，继续从原 CANN cache 查询 executor；新 shape、不同 extent、offset、格式、dtype、stream、设备或确定性设置重新计算。控制核数、不同标量或布局走原路径。输入值、输出、workspace 和 executor 均不放入线程单槽。实包 SHA256 为 `2f7c8c8786e2b2cb191f84a588e3b84ba8c1e40e9b706709357f4bb86f25a991`。[原生源](geometry.inc) · [对 pinned 源的修改](apply.py) · [完整编译收据](build-receipt.json)

正确性门分两次运行。一次每笔重新走原函数并核对记忆的 hash；另一次真正跳过重新序列化，只绑定新地址，与原 `div.out` 对照。两次都覆盖所有 60 个 shape、不同实际数据和地址、两个流、非零 storage offset、±Inf、NaN、非标准维度、稠密输入和其他除数。正式性能两腿仅 B 打开 `NPU_DIV_GEOMETRY_KEY`，计数与 hash 审计都关闭。诊断腿才打开 `NPU_DIV_GEOMETRY_COUNT`。若实际激活不足、hash 或输出不匹配、设备工作发生变化，不能归因为 Host 单次提交提速。[微门](micro.py) · [宿主调度和审计](run_locked.py)

## 双向 embedding 基线的实际空档

按原 Level1 的物理流、设备 Task ID 和 queue ID 筛出每步 1,024 个实际侧表 BroadcastTo。每笔将此前全流空档与同一个 autograd worker 的 `MeanBackward1`、嵌套 CPU 事件逐区间求交。第 4/5 步的实际 shape 为 [1,19] 和 [2,14]。

| 时段 | 第 4 步 ms | 第 5 步 ms |
|---|---:|---:|
| 1,024 组后继任务之前的全流空档 |87.321|85.087|
| 后继 MeanBackward1 入口之前 |31.369|16.821|
| MeanBackward1 入口至同一 div 入口 |11.367|11.067|
| div 入口至同一 enqueue 开始 |13.025|12.625|
| 其中 fresh 输出的 empty_tensor |5.112|4.938|

这些时段仍含同一时钟范围的其他线程工作，不能把全部交集视作候选可回收收益。第 4/5 步供 hash 及其他 producer 代码共享的可见余额为至多 7.913 / 7.687 ms。MeanBackward1 入口前有 13.711/7.961 ms 的 worker 无可见 CPU scope，evaluate_function: AddBackward0 独占 6.629/5.868 ms，AccumulateGrad 本体独占 6.449/0.273 ms。线程外或未打点的时间保留为未归因。[逐笔复算](trace_budget.py) · [CPU/空档交集](trace-budget.json)

## 尝试无优化路径

同一 ELF、同一 logical NPU 8 和同一 eager 操作边界，无 profiler；每腿在新进程内先预热，再取 9 个窗口的中位数。两个单形状窗口各做 256 个调用，60 形状窗口按原顺序每形状 16 个调用。每窗仅在首尾同步。下表单位为 µs/调用，正反序均按 A→B 报告。

| 调用形状 | 正序 A | 正序 B | 正序 B−A | 反序 A | 反序 B | 反序 B−A |
|---|---:|---:|---:|---:|---:|---:|
| [1,19] |14.570|15.382|+0.812|14.568|16.339|+1.772|
| [2,14] |14.582|15.192|+0.610|14.305|15.411|+1.106|
| 60 形状 |13.791|14.624|+0.833|13.568|16.322|+2.754|

两种语义模式均通过 64 组正例和 5 个回退。每种模式有 64 个新几何、128 次重复。其中 hash 审计 128/128 次等于原键；实际地址绑定模式的 BF16 输出和独立存储也逐项相同。两种模式各有 65 次 CANN executor hit、127 次 miss。首次脚本设定了缺少依据的“至少 100 次 hit”断言，原始 69 组数值和 128 次原键已经相同；修正为真实 hit 存在且 hit+miss 完整覆盖 192 个目标请求后，重新执行全部流程。三类同步窗口在两种顺序中全部回退，按预设门限停于微测，训练 E2E 和 Level0 Free 均未运行。[逐窗及门禁](micro-evidence.tgz) · [两腿计数和时间](micro-final.json) · [早停判据](performance-gate.json)

发布补丁从编译收据列出的 pristine 4 个单元与实际 stage 中的 6 个文件导出，在独立副本先执行 `git apply --check`，再将逐文件 SHA-256 与构建收据核对。补丁包含共同的双向 embedding 前置。[完整原生补丁](from-stock.patch) · [应用后 SHA-256](patch-verification.json)
