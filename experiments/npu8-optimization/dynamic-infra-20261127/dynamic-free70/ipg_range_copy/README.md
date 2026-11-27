## ZeRO IPG 归约桶一次提交相邻梯度复制

独立 stock Level1 覆盖同形状的首见和再见。主线程发往物理流 44 的 `aten::copy_` 在每个观察步均对应 1,052 次 enqueue、1,052 次异步 memcpy 和 1,052 个 SDMA。下表列该组 SDMA 之前所有流都无设备工作时段之和，以及该组各笔请求的 p95。ACL 列为该请求中可见 ACL API 的合计。四组观测的训练步号和参数状态不同。前导空档还含其他 Host 工作、流依赖和设备排队，实际节省量需用处理腿核实。[首见逐步统计](stock-source-census.json) · [再见逐步统计](repeat-source-census.json)

| 形状与步号 | SDMA 前全流空档 ms | ATen 入口至 enqueue µs | enqueue µs | 队列等待 µs | consumer 服务 µs | ACL µs |
|---|---:|---:|---:|---:|---:|---:|
| [1,19] 首见 #4 | 66.612 | 22.85 | 7.33 | 2.67 | 10.76 | 5.13 |
| [1,19] 再见 #12 | 80.560 | 32.37 | 10.86 | 9.37 | 32.63 | 11.07 |
| [2,14] 首见 #5 | 76.482 | 29.61 | 9.29 | 7.41 | 27.19 | 7.95 |
| [2,14] 再见 #13 | 83.994 | 31.08 | 13.65 | 7.50 | 26.85 | 7.05 |

D04 动态 Level1 中每个有效步的物理流 44 都出现了 1,052 次连续、同类的 SDMA。另有 1,052 个流 47 的单笔复制，多数与其他工作穿插。对两腿第 4、5 步逐个对齐 enqueue/dequeue correlation ID：当 consumer 开始服务队首时，同一连续段的下一项均尚未结束 enqueue。就可见的 enqueue 完成时间而言，等现有队列自然积累再合并可节省 0 次请求。观察界限为 enqueue 返回；真实指针发布稍早，需专用打点进一步定位。[逐步统计](queue-study.json) · [复算](queue_opportunity.py)

在原 ZeRO-2 归约端，`reduce_ipg_grads` 已经持有完整的 `bucket.params`。原方法完成 `average_tensor` 后，进入 reduction stream，对每个参数检查状态、调用 `copy_grads_in_partition`、将 `param.grad.data` 指向对应新分片。候选在该循环前直接验证已就绪的真实源、顺序和范围，将完整复制交给当前 PTA `aten::copy_` 一次。业务模型、反向 hook、归约和 overflow 使用原实现。开关设在 ZeRO 与 PTA 接口；改动位于训练框架接入层，使用现有的单笔 PTA 提交。[上游 reduce_ipg_grads](https://github.com/deepspeedai/DeepSpeed/blob/v0.19.0/deepspeed/runtime/zero/stage_1_and_2.py#L1547) · [运行时守卫和提交](range_copy.py)

只处理单卡 BF16 ZeRO-2、单参数组、非 offload、无轮转、每步一次反向、整个 IPG 桶涵盖本分区，且原 `reduce_ipg_grads`、`copy_grads_in_partition` 和 `get_gradient_for_reduction` 源码固定。每个源必须在同一完整 IPG storage 上，其逻辑片段依序连续、无空洞、无重叠，dtype/device/shape、参数编号和已归约标志符合原实现。目标为刚分配的独立连续缓冲。输入与目标在实际 reduction stream 调用 `record_stream`；原 `average_tensor`、依赖和后续同步保留。先建齐目标视图后调用一次 `copy_`，再按原序发布梯度视图、状态和 offset，并补齐原目标版本计数。不能满足条件时在写入前走原循环；已进入设备提交发生错误时让原异常上抛。实验分支需要改用通用 CANN batch API 时，应另验证非连续区间的顺序、地址数组生命周期及真实设备任务数。

CPU 独立门包括多尺寸、非零起始偏移、非有限值、1,052 个梯度、逐字节结果及别名、offset、标志和目标版本；负例覆盖逆序、不同 storage、已归约、短桶和已有目标。NPU 的同一套门另保留跨分配流的观察，以及错误流的回退。激活证明以真实训练结果 `rank000/ipg_range_copy.json` 为准，未运行的性能不能由 micro 推算。[CPU 结果](cpu-gate.json) · [专项门](micro.py)

在原容器中运行 `python micro.py --device npu --out <微测结果文件>`。在宿主将本目录复制到原实验 `dynamic-free70/ipg_range_copy`，以唯一名称将 `recipe.json` 复制至 `dynamic-free70/runner_tools/recipes/ipg-range-v1.json`。两腿沿用只启用 01、原分区和原 norm 的基线。

```sh
t=dynamic-free70/runner_tools
r="$t/recipes/ipg-range-v1.json"
w=/data/recsys-gr-ds019-host88/artifacts/dynamic-infra-20261127/20261127T033119Z-baseline-l1-341x7gd3/workspace
python3 "$t/run.py" "$r" gate --workspace "$w"
python3 "$t/run.py" "$r" l1/ab no-profiler/ab no-profiler/ba l0/ab l0/ba
python3 "$t/report.py" "$r"
```

证伪条件包括正式负载的实测源未连续、每步未出现一次覆盖全桶的命中、梯度位值或更新发生变化、额外打包搬运抵消收益，以及目标 SDMA 在原流上并未减少。用实际 Level1 中的 `NPUIPGBucketRangeCopy`、`aten::copy_`、consumer 及物理流编号核对。若原有区间守卫回退，`attempts[].reason` 记录原因；可用 `observe` 对照原路径的归约与生命周期。
