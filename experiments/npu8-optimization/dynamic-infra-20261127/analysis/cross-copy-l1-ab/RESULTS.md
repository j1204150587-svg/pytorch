# D04 跨分配流 D2D 的逐笔下发核对

第 4、5 步首次运行实际形状 `[1,19]`、`[2,14]`。两腿均开启同流 D2D 直接复制（#42），同时给满足跨源分配流条件的 `copy_d2d_baseformat_opapi` 记标。A 调用 `aclnnInplaceCopy`，B 先调用 `recordOtherStreamCopySource` 保护源块，再调用 `copy_d2d_by_memcpy`。训练公共配置为动态 eager、01 分块 overflow、TQ Mode 1；每个配对的实验开关是 `NPU_EXPERIMENT_D2D_RECORD_SRC`。[补丁](../../cross_copy/from-stock.patch) · [A 命令](../../evidence/cross-copy-l1-ab/01-A/command.json) · [B 命令](../../evidence/cross-copy-l1-ab/02-B/command.json)

每步反向主线程的 1,052 个目标 `aten::copy_` 在 A、B 按线程内 ATen 序号一一对应。每次 marker 均包含一个入队，两腿均对应一次 `aclrtMemcpyAsync` 和物理流 44 上相同流内序号的一次 SDMA。autograd worker 的另外 1,052 次 #42 直接复制同样逐笔匹配，落在物理流 47。每腿每步还有 1,114 个由 `aten::copy_` 发起的无标记队列项，当前标记未细分各自的条件分支。[标记和流内配对](copy_guard_evidence.json) · [逐笔调用](marked_copy_calls.csv.gz)

| 目标 1,052 次复制的阶段（µs） | #4 p50 A → B | #4 p95 A → B | #5 p50 A → B | #5 p95 A → B |
|---|---:|---:|---:|---:|
| producer，ATen 入口到 enqueue 开始 | 25.27 → 5.49 | 51.61 → 15.69 | 25.90 → 4.57 | 51.16 → 35.05 |
| enqueue 标记范围 | 7.68 → 5.73 | 10.39 → 16.88 | 7.78 → 7.97 | 9.20 → 19.49 |
| enqueue 结束到 dequeue 开始的有符号时差 | 6.19 → 2.87 | 7.93 → 5.05 | 7.07 → 2.84 | 8.48 → 9.50 |
| consumer，dequeue 服务 | 17.10 → 6.44 | 22.43 → 10.00 | 19.53 → 6.59 | 24.84 → 16.56 |
| ACL memcpy API | 5.47 → 4.98 | 7.32 → 7.63 | 5.52 → 4.76 | 8.28 → 11.68 |

A 的每笔目标复制都有一次可见的 ACLNN workspace 准备，#4、#5 共计 10.863、11.599 ms。B 的对应次数均为零。producer 和 consumer 两个形状的 p50、p95 均缩短。第 5 步 B 的 ACL memcpy p99 从 10.69 升至 15.73 µs，最大值从 21.50 升至 143.14 µs，consumer 服务最大值从 66.07 升至 186.70 µs。[完整时延分位数](marked_copy_latency.csv)

| 上一个全设备计算或搬运结束，到目标 SDMA 开始的空档（ms，按步求和） | #4 A → B | #5 A → B |
|---|---:|---:|
| 目标 SDMA 前的 1,052 段空档总和 | 79.395 → 63.765 | 82.743 → 69.435 |
| 其中与该请求的 workspace API 相交 | 10.393 → 0 | 4.152 → 0 |
| 与该请求的可见队列等待相交 | 5.290 → 3.019 | 6.755 → 1.866 |
| 与 dequeue 开始至 ACL API 入口相交 | 6.823 → 1.275 | 9.301 → 0.798 |
| 在该次 ACL API 返回后 | 26.723 → 30.811 | 47.556 → 48.892 |

目标 SDMA 前的空档总量分别减少 15.630、13.309 ms。空档 p95 则从 109.42 到 98.86 µs、从 115.06 到 177.50 µs。按空档起点定位，第 4 步 A 有 581 次在进入 ATen 后但首个 workspace API 前开始，B 有 584 次始于目标 ATen 调用前。第 5 步 A 有 209 次始于 workspace、411 次始于 enqueue、171 次始于队列等待；B 有 742 次始于 ACL 调用已返回后。跨越多个阶段的空档按时间交集切开，最后一行仍包括设备调度等不可进一步分辨的等待。[每个阶段的交集](marked_copy_preceding_gaps.csv) · [空档起点计数](copy_guard_evidence.json)

全部物理流的工作与控制任务序列相同。第 4 / 5 步各有 15,093 / 15,094 次 kernel launch，两步分别都有 2,121 次异步、6 次同步 memcpy，2,127 个 SDMA 任务，14 个设备 EVENT_RECORD 和 8 个 EVENT_WAIT。Host 每步均有 8 次事件创建、14 次记录、8 次等待。Host 事件销毁、查询随步变化，销毁第 4 步 9 → 11，第 5 步 11 → 8，PTA 销毁队列项差值 +2、−3。[全步提交和控制任务](submissions_compare.json)

Level1 的全步 Free 第 4 / 5 步为 692.107 → 706.630、654.754 → 969.735 ms；Stage 为 745.514 → 763.219、705.144 → 1,022.729 ms；Computing 为 53.157 → 56.331、50.139 → 52.688 ms。第 4 步 Host 反向 476.030 → 438.125 ms，optimizer 113.639 → 168.433 ms。第 5 步前向、反向、optimizer 都变慢。这里定位到的是单次复制外围准备和服务、以及后继 SDMA 前的局部空档，整步结论使用独立正反序无 profiler 和 Level0 配对。[A 原始 CSV](../../evidence/cross-copy-l1-ab/01-A/step_trace_time.csv) · [B 原始 CSV](../../evidence/cross-copy-l1-ab/02-B/step_trace_time.csv)

在实验根目录复算：

```sh
python -B cross_copy/attribute.py evidence/cross-copy-l1-ab analysis/cross-copy-l1-ab
```
