# PTA consumer 的线程调度与设备空档

本诊断由动态 eager runner 启动逻辑卡 8 的单个 stock A 腿。训练沿用 01 分块 overflow、Task Queue Mode 1、单 rank、原分区和 norm。阶段计时包含观测开销，端到端吞吐和 Level0 Free 使用独立的正式配对。

~~~sh
python3 -B queue_sched/run.py --runner runner --private runner-sched-01 --result sched-01.json
python3 -B queue_sched/analyze.py <运行目录>/01-A <分析输出目录>
~~~

私有 runner 在新工作区副本的训练循环前发送 PID 和唯一的 `acl_thread` TID。宿主通过容器 PID 列表、NSpid、工作目录 inode 及线程名确认二者，再创建独立的 tracefs 实例。实例在启用事件前设置 `sched_switch` 的 `prev_pid == TID || next_pid == TID` 和 `sched_wakeup` 的 `pid == TID` 过滤条件。持久化的 CSV 只包含目标线程的切出、切入、唤醒、前态和时间戳。捕获完成后删除本轮实例，事件丢失或目标与 profiler 的 dequeue TID 不一致时复算停止。设备状态检查和宿主共享锁由同一个动态 runner 执行。

## 对时与归因

TorchNPU 导出的 CPU trace 使用转换后的时间轴。开启 syscnt 时，`ProfilerConfig.get_timestamp_from_syscnt` 先以 Host cntvct、频率和 `clock_monotonic_raw` 计算时间。随后 `CANNFileParser.get_localtime_diff` 使用 `collectionTimeBegin × 1000 − clockMonotonicRaw` 计算 ns 偏移。原始文件未保留时，下面的训练进程对时仍覆盖本轮实际导出的 CPU 事件。[TorchNPU 导出配置](https://github.com/Ascend/pytorch/blob/38e483ee79ec8a3286517808740a811c0f0ae66c/torch_npu/profiler/analysis/_profiler_config.py) · [CANN 时间转换](https://github.com/Ascend/pytorch/blob/38e483ee79ec8a3286517808740a811c0f0ae66c/torch_npu/profiler/analysis/prof_parse/_cann_file_parser.py)

采样使用私有实例的 `mono_raw`。每个活跃步有五个 `record_function`，训练主线程在各入口和出口两侧调用 `clock_gettime_ns(CLOCK_MONOTONIC_RAW)`。由实际导出的 marker 及其四个边界，取本次 trace 到 raw 时钟偏移的交集，导出时间量化留 ±0.5 µs。保留 CANN 的 `start_info` 时另交叉核对。状态转换两侧按该偏移区间和 tracefs 时间戳分辨率收缩，落在边界中的区间单列为未分辨。缺失唤醒时，睡眠和等待调度的分界同样单列。

`exposed_scheduler.csv` 中每行是一种算子、producer 角色和时间边界的毫秒数。全设备流上一段计算或搬运结束至后继实际设备工作的空档，按同一请求的 queue ID 联结。可见排队区间再按照同一 consumer 是否在前序 dequeue 内拆分。另统计消费开始到本次 ACL API、ACL API 自身与空档的交集。各范围内按以下观测解释线程状态。

- `scheduled` 表示 Linux 已将该线程调度上 CPU。此区间可能包含中断和内核态。
- `off_blocked` 由非 runnable 前态的切出至目标线程实际唤醒计量。`runnable_after_wakeup` 从该次唤醒到重新调度上 CPU。
- `runnable_after_preempt` 要求 `sched_switch` 的 `prev_state=R+`。`runnable_switch_out_R` 保留普通 runnable 切出的事件。
- 即使定位到睡眠，阻塞在 eventfd、其他系统调用或未标记代码的细分仍需额外观测。PTA 队列等待起点使用可见的 enqueue 结束，实际写指针发布的位置仍有差异。

原始调度事件在腿内的 `rank000/queue_sched/scheduler.csv`，同目录保存容器及宿主 TID、丢失计数和实测校准点。`calibration.json` 给出偏移区间和每个 marker 的界限。
