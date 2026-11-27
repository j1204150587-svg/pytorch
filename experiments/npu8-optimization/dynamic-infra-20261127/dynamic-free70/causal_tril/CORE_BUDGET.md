# 动态 eager 主干及尾部的预算

在 Direct B 的 Level0 中，以第一处主干 RMSNorm 的 `aten::pow` 为前向起点，止于交叉熵前。用这些前向 ATen 的 `Sequence number` 找到相应的 autograd worker 反向节点，再与官方 Free 轨取墙钟交集。#4–6 实际形状是 [1,19]、[2,14] 和 [2,22]。下表单位均为 ms。[逐步数据](direct-core.json) · [复算脚本](core_budget.py)

|窗口|#4|#5|#6|
|---|---:|---:|---:|
|主干前向长度|6.416|6.176|6.493|
|主干前向与 Free 交集|5.750|5.537|5.865|
|其中前向主线程 ATen 之外的时间上限|1.378|1.278|1.435|
|174 个已匹配主干反向节点的墙钟并集|10.329|10.105|9.996|
|这些节点与 Free 交集|9.445|9.170|9.069|
|其中节点内部、该 worker 的 ATen 之外的时间上限|2.320|2.190|2.191|
|4 层 causal 掩码前向相关 ATen 与 Free 交集|0.476|0.408|0.452|

被匹配节点之外仍有 autograd 调度空隙，时间上限也包含未标记的 C++ 工作。仅缩短已匹配节点里的 Python 入口和 autograd 外壳，能直接触及的前后向时间约 3.5–3.7 ms。以原生库复合算子包住一次整层 forward/backward 可以保留原 BF16 边界，进一步获益需要缩短 ATen 以内的参数准备、减少实际提交，或者覆盖其他阶段。

当前 Free 约 28.29 ms、Computing 约 3.50 ms/步。若其余 Stage 时间稳定在 3.5–3.6 ms，达到 Free 占比低于 70% 要求 Free 少于 8.2–8.4 ms。即使主干标记内约 14.95 ms 的 Free 全部消失，仍有约 13.34 ms。完整的 forward_and_loss / backward / optimizer 三个阶段各与 Free 相交约 10 / 12.4 / 4.9 ms。optimizer 的原生 norm 相关窗口就占约 3 ms，当前梯度打包 cat 约 0.063 ms。嵌套窗口不可相加，实测 Stage 与 Computing 也会随候选改变。

## 可保位值的实现和证伪点

1. 用 libtorch C++ eager 复合算子封装主干的真实逐算子调用，再实现手写一阶反向。首个最小切片覆盖每步 9 次的 64 宽 RMSNorm。复用同版本原生 pow、mean、add、rsqrt、mul、sum_to_size 和 BF16 参数，保持原来的舍入、规约、广播以及残差梯度累加顺序。HSTU 内 dropout 若启用，使用原始设备 RNG 和 mask 保存规则。先对齐输出、所有输入与权重梯度、一次正常更新以及注入 overflow 的跳步，再对前向及反向按 stream 核对真实 kernel/SDMA 序列。如果仅消除了若干 Python 节点、Free/E2E 的反序测量持平，就结束这一条。当前四类输入的原生 `npu_rms_norm` 已出现位差，必须独立作证。
2. 若 C++ eager 仍在 launch 前有持续空档，下一层需定位到可合并的 producer 准备或 Queue Mode 1 入队。检查目标 wheel 对应的 `OpCommand::RunOpApi[V2/V3]`、`Repository::Enqueue`、`Repository::Dequeue`。预先构造经过原始动态 shape、stride、offset、dtype、format 和 stream 校验的一串按序执行描述符，再一次性入队。仅在所有依赖已由 Tensor 元数据确定、各工作区按原规则持有且全链单流时触发。consumer 直接执行原 ACL phase2，不得从 consumer 再走可重新入队的 ATen 实现。保存引用直到原异步 release 点，遇旁路、同步、设备切换、事件、数据依赖形状和异常时回退原调用。producer 首条请求准备齐后再等待其余请求可能推迟首核，必须和原实际首核时间配对。
3. 对第二条先取与同一后继设备任务严格链接的 producer、queue、consumer、ACL 时间，并按实际形状区分首次和重复路径。若减少 enqueue 而真实设备提交相同，只能计为 Host 批处理；若源码层实际合并设备操作，则分别报告 kernel、SDMA 及各 stream 的减少量。发生队列积压或计算已覆盖 Host 时间时，API 计时缩短无法证明 Free 收益。禁止把前后向所有标记的交集作为可回收时间相加。
