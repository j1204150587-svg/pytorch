# 单 rank 标量 HCCL 的提交、官方 Free 与实验边界

每个观测步有两个单元素、单 rank 的 HCCL AllReduce。溢出标志对应 INT32，平方范数对应 FP32。DeepSpeed 溢出路径用 MAX、范数用 SUM，trace 汇报数据类型和计数，未直接导出 reduce-op 枚举。stock 另有多元素的 IPG 集合通信，以下只计标量。通过同一线程的 c10d scope、PTA enqueue/dequeue correlation_id、CANN 的同起止 Node@launch、communication 上相同 connection_id，以及硬件 task 的 connection_id 逐笔联结。5 个独立原始 Level1 共 20 笔标量调用，每笔均在物理流 45 出现两次 4 字节 Reduce_Inline/SDMA。[逐笔数据与区间交集](trace.json) · [复算](trace_scalar.py)

下表给出 HCCL Host API `AscendCL@hcom_allReduce_` 执行范围与官方 `Overlap Analysis/Free` 时间段的交集，单位 µs。同一步两类标量的 API 没有重叠。此交集与设备关键路径有关联，不能保证能全部回收。

| 配置 | #4 INT32 | #4 FP32 | #4 合计 | #5 INT32 | #5 FP32 | #5 合计 |
|---|---:|---:|---:|---:|---:|---:|
| stock 01 | 155.705 | 140.545 | 296.250 | 338.475 | 162.375 | 500.850 |
| direct A | 76.950 | 63.490 | 140.440 | 85.130 | 82.890 | 168.020 |
| direct B | 80.300 | 80.110 | 160.410 | 70.770 | 76.090 | 146.860 |
| fold A | 95.190 | 73.420 | 168.610 | 78.010 | 68.300 | 146.310 |
| fold B | 90.600 | 102.060 | 192.660 | 87.030 | 73.230 | 160.260 |

在 direct B 的 #4/#5，INT32 的 HCCL Host API 分别长 115.24/114.69 µs。前序工作为流 45 的 Cast；HCCL 第一个 SDMA 前的全流空档与 API 交叠 57.09/51.52 µs，最后一个 SDMA 至后续 Cast 的全流空档与 API 交叠 23.21/19.25 µs。FP32 的前序为流 47 的 ReduceSum，后续为同流 Pows。stock 的 FP32 请求还分别等待队列 1.109/1.991 ms；这两段发生在消费者进入 HCCL 前。官方 Communication 区间、单笔 HCCL 设备任务时间、Host API 的 Free 交集分别保存在原始分解中。若去掉设备通信，原先计入 Communication 的时段可能转为 Free，必须另用 Level0 判定整步占比。

## 主机提交流程与可检验的改动

在固定 PTA 提交 `38e483ee79ec8a3286517808740a811c0f0ae66c` 的 `ProcessGroupHCCL::allreduce`，单 rank 的整数 MAX 满足原地取最大值的恒等条件。实验 B 仅处理组大小 1、rank 0、TQ1、非 capture、普通独立 allreduce、无 coalescing/OpHook/desyncDebug、一个原地连续 INT32 标量、HCCL 有效元素数 1、原请求和转换后枚举均为 MAX、输入输出地址相同。保留 HCCL 通信器初始化、异步队列项、allocator 记录、前后跨流 event、bool/byte 的前置 cast 和结果回写、Work/Future、MSTX。仅在原消费者 lambda 中确认 `HcclGetCommAsyncError` 成功且状态健康后将原始 `HcclAllReduce` 替换为该标量的已派发状态。其他调用及健康检查失败均走原路径。该试验的 HCCL watchdog 和正常 Work/Future 错误路径仍由原框架管理；仍需实机验证健康查询成本及跨流可见性。

本版本以 INT32 MAX 为候选，FP32 SUM 在原路径运行。stock 微测另外覆盖正负零、两种次正规值、Inf、安静 NaN 和信号 NaN，按原始 FP32 字比较。实机发布新库前对 A 和 B 分别运行同一个 `micro.py`，核对相同输出、Work.wait、跨流依赖和 Future，B 中仅三个整数/byte/bool MAX 命中。继续训练前须确认八实形状逐步梯度、overflow 跳步和权重一致。同一步 A/B Level1 应保留压缩成单标量的 c10d 与 PTA 回调和事件，B 比 A 少一笔真实 HCCL Host 提交及流 45 的两个 Reduce_Inline。端到端和全步 Free 的判定分别用无 profiler 及独立 Level0。[隔离构建](build.py) · [原生通信门](micro.py) · [训练配方](recipe.json)
