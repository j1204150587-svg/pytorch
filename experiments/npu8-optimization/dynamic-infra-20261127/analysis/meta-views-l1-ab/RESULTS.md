# D03 修订跳过权重恢复的重复 Host 视图构造

Level1 正序 A/B 均在第 4 步 `[1,19]`、第 5 步 `[2,14]` 比较。每步 A 的权重恢复进入一次 `aten::unflatten_dense_tensors`，B 进入一次 `npu_dynamic_meta_views::try_unflatten` 并命中 `NPUDynamicMetaUnflattenViews`。大组有 1,052 个 BF16 Meta 模板。每步省去 `narrow`、`slice` 和 `view` 各 1,052 次，保留 1,052 个 `as_strided`。另一个单模板 NPU allreduce 小组两腿均走原 helper，四种 ATen 操作各一次。[A 方案与开关](../../evidence/meta-views-l1-ab/01-A/command.json) · [B 方案与开关](../../evidence/meta-views-l1-ab/02-B/command.json) · [原生实现和适用条件](../../meta_unflatten/README.md)

| 权重恢复，ms | 第 4 步 A → B | 第 5 步 A → B |
|---|---:|---:|
| 原生大组父 scope | 19.712 → 4.506 | 25.191 → 5.526 |
| 父 scope 与全部物理流上设备工作的交集 | 0 → 0 | 0 → 0 |
| 最后一个设备工作任务至 Host 步末 | 31.801 → 17.486 | 42.219 → 20.276 |
| 最后一个设备工作任务至下一步首个工作任务 | 33.782 → 20.413 | 超出采集窗 |

第 4 步 A、B 的大组 scope 都完全落在官方 Free 中。其前为物理流 47 的 `aclnnInplaceCopy_CastAiCore_Cast`，其后为第 5 步同流的 `aclnnArange_ArangeAiCore_Range`。两者之间的空档缩短 13.369 ms，比原生 scope 的缩短量 15.206 ms 小。前一设备核结束至原生 scope 入口由 0.356 增至 1.087 ms，原生 scope 退出至下一核开始由 13.714 增至 14.820 ms。这两段覆盖 Python 入口、其他 CPU 工作及两端的设备调度。第 5 步位于 trace 末端，官方 Free 轨在最后一个设备任务处截断，无法由该轨测得该步回写窗口的 Free。[A 逐 scope 分解](A/drain_and_tail.csv) · [B 逐 scope 分解](B/drain_and_tail.csv)

第 4 / 5 步剩余 `as_strided` 的 p50 分别为 1.26→2.23 / 1.81→2.05 µs，p95 为 10.26→7.63 / 16.92→13.67 µs，p99 为 16.71→17.82 / 20.27→20.15 µs。每步 1,052 笔该子操作的时间和由 2.963→3.374 / 3.958→4.374 ms。A 中 `narrow` 包含 `slice` 和 `as_strided`，嵌套的时长不能相加。父 scope 的改进来自减少重复的 Host 操作。Python 适配器对每个调用的检查在新原生 scope 之外。[子操作各分位数和最大值](B/drain_and_tail.csv)

两腿物理流 44 / 45 / 47 的实际工作数在第 4 步均为 1,053 / 8 / 16,159，在第 5 步均为 1,053 / 8 / 16,160。各流内逐槽核名、任务类别及包含硬件控制事件的序列相等。两步的 `aclrtLaunchKernelWithHostArgs` 分别有 15,093 / 15,094 笔，同步 / 异步 memcpy API 分别为每步 6 / 2,121 笔。第 5 步 B 多 1 笔 Host 队列 `destroy_event`，同一步物理设备任务序列相同。整个 Level1 窗的 Free 为 851.255→733.811 / 803.351→752.134 ms，差值超过本局部 scope 可以解释的量。标准 Free 和端到端需分别看双序 Level0、无 profiler 窗口。[逐物理流与序号](physical_stream_substitutions.json) · [全步 ATen、排队与 ACL](submissions_compare.json) · [A 官方 CSV](../../evidence/meta-views-l1-ab/01-A/step_trace_time.csv) · [B 官方 CSV](../../evidence/meta-views-l1-ab/02-B/step_trace_time.csv)

训练语义门的 8 步包含 3 次非有限跳步和 5 次权重更新。每步比对 1,052 组梯度，norm 差为 0，最大 loss 绝对差为 7.63×10⁻⁶。B 共命中 5 次权重恢复，8 次单模板回退。独立微测核对了非零基址、尾部 padding、别名、正反 AD、空与异类模板、原越界错误文本和新装显式 AutogradMeta 后的回退。[训练语义](../../evidence/meta-gate/comparison.json) · [B 微测](../../evidence/meta-micro/02-B/micro.json) · [后注册分派验证](../../evidence/meta-override/01-B/micro.json)

在上一级 `analysis` 目录复算。

```text
python -B compare_dynamic_legs.py ../evidence/meta-views-l1-ab/01-A ../evidence/meta-views-l1-ab/02-B meta-views-l1-ab
python -B stream_substitutions.py meta-views-l1-ab
python -B inspect_tail.py ../evidence/meta-views-l1-ab/01-A meta-views-l1-ab/A
python -B inspect_tail.py ../evidence/meta-views-l1-ab/02-B meta-views-l1-ab/B
```
