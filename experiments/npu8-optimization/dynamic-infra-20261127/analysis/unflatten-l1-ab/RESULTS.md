# D03 初次 Level1 配对未激活

#4 `[1,19]`、#5 `[2,14]` 的权重回写均进入了 `aten::unflatten_dense_tensors`。B 启动时设置了 `NPU_EXPERIMENT_UNFLATTEN_VIEWS=1`。两个步骤 A/B 的大组 scope 均有 1,052 次 `narrow`、1,052 次 `slice`、1,052 次 `as_strided` 和 1,052 次 `view`，四笔大组 scope 的 `NPUUnflattenContiguousViews` 标记均为零。正常命中应只剩每片一次 `as_strided`。同一步另有一次单模板 unflatten，双方均执行普通路径。该配对未触发设计修改。[B 腿启动参数](../../evidence/unflatten-l1-ab/02-B/command.json) · [A 的逐 scope 数据](A/drain_and_tail.csv) · [B 的逐 scope 数据](B/drain_and_tail.csv)

| 第 4 / 5 步，ms | A | B |
|---|---:|---:|
| 权重回写大组 unflatten 墙钟 | 24.013 / 21.278 | 9.132 / 24.083 |
| 大组 scope 内全部设备流工作交集 | 0 / 0 | 0 / 0 |
| #4 上一步末任务到下一步首个物理任务 | 44.686 | 19.348 |
| #5 末工作任务至 Host 步尾 | 32.492 | 39.257 |

两腿 #4 的大组 unflatten 都落在官方 Free 中。其前一任务是物理流 47 的 `aclnnInplaceCopy_CastAiCore_Cast`，其后一任务是同流、下一步的 `aclnnArange_ArangeAiCore_Range`。从前一任务完成至 unflatten 开始为 0.286 / 0.335 ms（A/B），unflatten 结束至下一任务开始为 20.387 / 9.880 ms。#5 是最后活跃步，末核之后的 Free 和下一步设备首任务超出当前 trace 范围。

全步 ATen 计数和逐线程序列相同。流 44、45、47 的工作任务（含计算与搬运）、control event 及全任务顺序逐槽相同；#4 的 17,220 个和 #5 的 17,221 个真实工作任务无增减。两腿 ACL 各类提交数相同。#5 的 `destroy_event` Host 队列项 B 比 A 多 2 个；硬件事件序列在该窗口相同。Level1 Free 虽从 895.437→778.088 / 776.896→697.913 ms，这两步没有激活证据，不能用于认定候选获益。需以实际加载库、注册入口、真实 flat 与模板守卫信息区分安装、分派或守卫回退，后续修复配对应独立记录。[各物理流双向匹配](physical_stream_substitutions.json) · [算子、queue、ACL 汇总](submissions_compare.json) · [A 原始 CSV](../../evidence/unflatten-l1-ab/01-A/step_trace_time.csv) · [B 原始 CSV](../../evidence/unflatten-l1-ab/02-B/step_trace_time.csv)

在上级 analysis 目录复算。

```text
python -B compare_dynamic_legs.py ../evidence/unflatten-l1-ab/01-A ../evidence/unflatten-l1-ab/02-B unflatten-l1-ab
python -B stream_substitutions.py unflatten-l1-ab
python -B inspect_tail.py ../evidence/unflatten-l1-ab/01-A unflatten-l1-ab/A
python -B inspect_tail.py ../evidence/unflatten-l1-ab/02-B unflatten-l1-ab/B
```
