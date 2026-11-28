# 后继物理任务开始前，两个 producer 在做什么

原双向 embedding B 的 Level1 第 4、5 步共记录 17,219 和 17,220 个全流空档，合计 429.845 和 424.379 ms。设备时钟按 trace 原样与 Host 对齐时，后继 ATen 尚未开始、两个 producer 又都处在可见实际提交操作以外的时间为 **101.357 和 69.580 ms**。其中与其他请求的任何 consumer dequeue 区间重合的只有 0.651 和 0.240 ms。在同一个时间点，缩短后继请求自身的 producer 参数准备或 consumer 执行，无法直接消除它尚未进入 ATen 的这段等待。提前完成更早的依赖、改变训练线程调度或提前提交后继工作，可能移动整个时间线。[时间线复算](coverage.py) · [全部物理空档](physical-gap-coverage.csv.gz)

## 互斥分解

每个空档为全设备所有算术、拷贝与通信工作区间并集中的洞，从前一项工作的最晚结束到下一个物理任务开始。只有完整落入已记录 step 的间隙参与。边界外的首尾等待和同步不计入这里，Level0 Free 宜独立读原来的第 4–6 步数据。下表单位 ms，主表三类在每个 offset 下相加都等于相应的单步总量。区间列给出满足时钟约束的最小值和最大值。

| 后继物理任务所处阶段 | #4 [1,19] 名义 | #4 全时钟范围 | #5 [2,14] 名义 | #5 全时钟范围 |
|---|---:|---:|---:|---:|
| 后继 ATen 尚未进入，两个 producer 均未处在有实际提交的 ATen 或直接 enqueue/API 调用内 |101.357|34.552–104.816|69.580|33.632–99.453|
| 对应本笔物理任务的 ACL launch/memcpy 已返回 |133.801|123.699–387.313|234.939|117.611–384.489|
| 其余阶段及无法唯一关联的后继 |194.688|7.980–201.330|119.861|6.258–207.314|
| 全流空档总和 |429.845|429.845|424.379|424.379|

对于“后继请求已提交”，Host 与设备之间有两层边界。名义时钟下，PTA **已接受入队**的总覆盖为 227.895、296.876 ms（含上表 ACL 已返回）。以 enqueue 返回、consumer 开始两者的较早时刻为“最迟已接受”的可观察时刻，避免把返回前已被 consumer 取走的请求记为未发布。真实环形队列写指针发布发生在该时刻或更早。名义时钟下其余阶段的细分如下。

| 其余阶段，无重复计数 | #4 [1,19] ms | #5 [2,14] ms |
|---|---:|---:|
| 后继 ATen 未进入，另一个可提交 op 正在 producer 上 |0.235|0.153|
| 缺后继 ATen scope，尚未到 enqueue |0.119|0.091|
| 后继 ATen 已进入，尚未到 enqueue |71.614|37.515|
| enqueue 正在执行，尚未到最迟已接受时刻 |27.593|19.453|
| 已入 PTA 队列，consumer 尚未开始 |18.466|12.460|
| consumer 已开始，实际 ACL 提交 API 尚未进入 |27.752|19.747|
| 实际 ACL 提交 API 进行中 |47.877|29.730|
| 每步 6 笔具有 Node@launch、缺少唯一 consumer/API 的 MEMCPY_ASYNC |1.032|0.712|

在允许的校时范围中，即使取最保守值，每步仍至少有 123.699/117.611 ms 满足“同一物理任务的 ACL API 已返回”，至少有 34.552/33.632 ms 满足主表第一类。两类都是时间顺序的下限。异步命令的生效时点、stream 事件依赖、固件调度和设备内部间隙仍可能影响第二类。不能用其大小直接估算任何单项代码改动的收益，也不能从中断言所有 Host 工作已离开关键路径。

## producer 的判定

绝大多数入队项来自主线程 334627 或 autograd worker 334677。第 4 步第三个线程 334654 仅有六个 `destroy_event`；两个 step 内 34,427 个有唯一完整链路的物理空档，其后继均来自前两条线程。逐笔从硬件 flow 和 connection ID 关联 CANN anchor、实际 ACL API、dequeue、enqueue 和同线程最内层完整包住 enqueue 的 ATen。一个队列项拆成多笔设备任务时，每笔取自己的真实 ACL API。其余 12 个空档原样保留。

主表以两个 producer 的“实际包含 enqueue 的最内层 ATen 的完整区间，加所有直接 enqueue/API 时间”取并集。未观测到提交不代表一个 op 的所有实际输入都不会提交。将口径扩大到“本 trace 中曾直接或间接含 enqueue 的所有 ATen 名字的每次执行”，主表第一类缩为 96.836/65.061 ms。若要求两线程连任何已记录的 ATen 都没有，则仅余 **75.437/49.469 ms**（校时全范围 25.987–77.892/25.104–71.912 ms）。这两个数都是主表第一行的子集。无 CPU scope 期间可能运行 Python、autograd C++、未打点运行库、系统调用或被换出。[逐步及敏感性数据](coverage.json)

## 可继续定位的时间

主表第一类中，按下一笔工作的 producer 上实际执行的最内层 CPU scope 做互斥计数。两个 step 的主要位置如下。CPU scope 和后继物理任务可能归属不同层级，以下数值都是同一时间区间的相交值。

| 命名范围或后继 | #4 [1,19] ms | #5 [2,14] ms | 可定位到的边界 |
|---|---:|---:|---|
| `torch::autograd::AccumulateGrad` |17.047|9.473|其中后继 Mul 为 9.392/8.410 ms。scope 包含 autograd 和注册的 hook。|
| 后继 producer 无 CPU scope，处在观测时间线内 |14.278|9.644|其中后继 Divs 为 13.723/7.980 ms。仅凭此 trace 无法区分执行、等待与被抢占。|
| `aten::view/as_strided/unsqueeze/expand/slice` 的互斥区间合计 |21.173|16.505|含 view 创建、autograd 元数据和公共 ATen 路径。PTA view 元数据分配已另作候选，完整 scope 的持续时间是这一类同刻可见的上界。|
| `evaluate_function: AddBackward0` |6.645|5.881|其中后继 Divs 为 6.631/5.873 ms。|
| `deepspeed_backward` 最内层已命名范围 |10.382|7.089|后继分散在 Copy、InplaceDivs 等，缺少命名到具体 Infra 函数的自耗时。|
| 优化器 step 最内层已命名范围 |7.237|7.069|其中后继 InplaceZero 为 4.909/4.431 ms。|
| `npu::get_npu_format` |1.326|1.328|这里的区间较小。|

在这部分空档内，44.727/30.089 ms 后继是 Divs，14.292/12.660 ms 后继是 Mul。下一笔 Divs 尚未进入 ATen 时，优化该笔 Divs 内部的 hash、workspace 或 ACL 入口只能影响之后的时段。可见的十毫秒量级区间集中在 autograd、训练 hook、CPU view 与未命名时间，现有证据没有识别出区别于已测方向、每步稳定暴露数十毫秒的单个 TorchNPU/op-plugin/CANN 函数。若追查未命名区间，最具辨识力的下一项观测是把同一 rank 的两条 producer 的运行/换出、调用栈、autograd hook 和命名 scope 对齐到具体物理任务。当前回溯到的是候选定位范围，未获得可归因的新优化收益。[已有 view 方向](../followup70/native-host/README.md) · [已有反传 Divs 门](../div-geometry/README.md)

## 时钟和复算

原 trace 文件 SHA256 为 `a37a06f9c4395e9484eba9f180e5ed90a48736e6770e654c47d42d0fdf064e2b`。先从同一 trace 运行主控的 `clock_bounds.py`，得到与其 `baseline-clock.json` 一致的 `device_time + δ` 约束。每步的最早设备时间满足 API 进入早于设备任务开始，最晚设备时间受全设备 drain 返回约束。#4 的 δ 为 −1.090 至 +167.170 µs，#5 为 −15.040 至 +148.550 µs。保留原 step 和原物理后继，校时仅移动间隙两端，使用整个 trace 的 producer/consumer 时间线；首尾跨界不会改换样本。对各个 host 时间段 [a,b] 与移动间隙 [l+δ,r+δ] 的交集，遍历 a−r、b−r、a−l、b−l 四类拐点，取得同一 step 内所有任务共用一个常量偏移时的精确极值。三类与各子项的极值可能对应不同偏移，不能直接相加。[时钟约束程序](../followup70/clock_bounds.py) · [同源复算结果](clock.json) · [主控交叉检查](../followup70/baseline-clock.json)

在本目录执行下列命令。第一条只依赖固定的输入 trace，第二条读取已有逐笔工作映射、原 trace 和第一条的时钟结果，生成本报告的 JSON 与 CSV。

```bash
python -B ../followup70/clock_bounds.py ../evidence/raw/20261128T043752Z-ab-l1-7m6ais34/02-B/trace_view.json.gz ../../dynamic-infra-20261127/analysis/analyze_dynamic_trace.py clock.json
python -B coverage.py
```
