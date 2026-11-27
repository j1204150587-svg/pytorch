# 逻辑 NPU 8 的 PTA consumer 线程调度实测

2026-11-27，动态 eager、01 分块 overflow、Task Queue Mode 1、1,024 张侧表、原生梯度分区和 norm。独立 Level1 第 4、5 步分别首次使用 `[1,19]`、`[2,14]`。两个步骤的官方 Stage 为 832.658、792.777 ms，Free 为 777.494、741.272 ms。诊断腿额外采集线程调度事件，这些步时只用于归因。[启动命令](../../evidence/scheduler-l1/01-A/command.json) · [官方 CSV](../../evidence/scheduler-l1/01-A/step_trace_time.csv)

**在已经进入 ACL 提交 API 的设备空档里，consumer 至少 99.5% 的时间已被 Linux 调度上 CPU。** 在前序 dequeue 之外等待的阶段，两个步骤都能确认一部分实际睡眠及醒后等待 CPU。其余微秒级时间片的边界仍有较大的对时不确定性。

| 后继请求与设备空档重合的阶段 | #4 空档交集 | #4 可确认已在 CPU 上 | #5 空档交集 | #5 可确认已在 CPU 上 |
|---|---:|---:|---:|---:|
| 可见排队期间 consumer 仍在前序 dequeue | 6.266 ms | ≥6.189 ms | 0.457 ms | 0.457 ms |
| 可见排队期间 consumer 在前序 dequeue 外 | 32.952 ms | ≥5.848 ms | 19.068 ms | ≥1.067 ms |
| 后继请求开始 dequeue 至本次 ACL API 入口 | 84.266 ms | ≥79.125 ms | 54.690 ms | ≥50.987 ms |
| 本次 ACL 提交 API | 65.812 ms | ≥65.523 ms | 29.803 ms | ≥29.784 ms |

表内按真正结束空档的设备任务匹配其 Host API 和队列请求，截取跨所有设备流的工作空档。consumer 已在 CPU 上仍可能处于内核态或遇到中断。可见排队从 enqueue 结束计时，真实入队写指针发布时间仍未打点。[按步、阶段的时间上下界](common_offset_bounds.csv) · [逐个设备空档及其队列关联](links/device_gaps.csv.gz)

## 前序 dequeue 外的等待

同一 trace 采用一个经校验的共同偏移。下表逐状态取所有可行偏移上的最小、最大交集，并纳入 tracefs 的时间分辨率。不同状态的极值可能出现在不同偏移，不能相加作为同一次调度的时间分布。

| consumer 线程状态 | #4 的 32.952 ms | #5 的 19.068 ms |
|---|---:|---:|
| 非 runnable 切出至 `sched_wakeup`，实际阻塞 | 2.279–23.097 ms | 1.108–15.656 ms |
| 唤醒至切入 CPU，已 runnable | 0.367–13.434 ms | 0.391–8.822 ms |
| 以普通 `R` 状态切出后等待 CPU | 0.158–0.267 ms | 0.231–0.275 ms |
| 已在 CPU 上 | 5.848–27.093 ms | 1.067–15.019 ms |

第 4 步的主线程 `InplaceCopy` 对应其中 11.287 ms，第 5 步的 worker `EmbeddingDenseBackward` 对应其中 8.897 ms。按逐时间片保守归因，这两组分别至少有 0.889、0.573 ms 可确认阻塞。整段捕获记录 36,678 次切出，其中普通 `R` 为 512、`S` 为 35,997、`D` 为 169。该内核此次没有输出 `R+`，普通 `R` 的成因尚不能细分。睡眠原因究竟是 eventfd 还是其他阻塞点，需要等待点和写指针标记。现有证据支持把绝大部分 ACL API 内的调度等待排除为主要瓶颈，对唤醒或轮询改动的端到端收益尚无直接配对。[按算子、线程及保守可辨时间切分](exposed_scheduler.csv) · [实际调度事件](../../evidence/scheduler-l1/01-A/rank000/queue_sched/scheduler.csv)

## 目标与对时

独占实验使用的私有 tracefs 实例在内核过滤目标 consumer，宿主 PID/TID 为 829818/963008，训练容器内为 59964/59989。Level1 的所有 dequeue 均来自对应容器 PID/TID。本腿取得 109,993 条调度事件，per-CPU 环形缓冲区 overrun、commit overrun、dropped events 增量均为 0，保存的调度时间戳分辨率为 1 µs。[采样回执](../../evidence/scheduler-l1/01-A/rank000/queue_sched/scheduler.json)

每步 5 个训练 marker 的起止点都使用 `CLOCK_MONOTONIC_RAW` 括住，20 个边界得到共同偏移区间宽 5.340 µs。TorchNPU 的路径选择器在有 `host/start_info` 时优先使用该文件。按原始字段计算，Host 偏移为 1,794,821,285,636,435,890 ns，落在 `[1,794,821,285,636,434,240, 1,794,821,285,636,439,580]` ns 内。备用的 `device_0/start_info.0` 独立采样偏移早 9.490 µs，本次 exporter 采用 Host 文件。[对时结果及全部边界](calibration.json) · [Host 采样](../../evidence/scheduler-l1/01-A/rank000/profile/clock-source/start_info) · [设备备用采样](../../evidence/scheduler-l1/01-A/rank000/profile/clock-source/start_info.0) · [TorchNPU 选择路径](https://github.com/Ascend/pytorch/blob/38e483ee79ec8a3286517808740a811c0f0ae66c/torch_npu/profiler/analysis/prof_common_func/_path_manager.py#L97)

从实验目录复算。

~~~sh
python -B queue_sched/analyze.py evidence/scheduler-l1/01-A analysis/scheduler-l1
~~~

`exposed_scheduler.csv` 仅将每一时间片在所有可行偏移下都一致的部分归入已知状态。`common_offset_bounds.csv` 按每种状态求整步总时间的上下界，先按 1 µs 量化误差收缩或扩展各段，再在整段 trace 的共同偏移区间内扫描相交时长函数的折点。[复算入口](../../queue_sched/analyze.py) · [总量上下界算法](../../queue_sched/shift_bounds.py)
