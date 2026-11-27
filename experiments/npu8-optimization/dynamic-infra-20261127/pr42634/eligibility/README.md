# !42634 在原动态 HSTU 的触发范围

PR 合入 v2.10.0，源分支末提交为 `2701f7b3e8d89625752edbde10e42c2b2172272c`。入口为 NPU Inductor 的 `NPUPythonWrapperCodeGen` 生成的 Triton 调用点。普通 autotuner 稳定后通过 `BoundFastLaunch` 进入 `_npu_inductor_fast_launch_with_plan`。grouped autotuner 绑定其原 `run`。原 PR 在 `torch.autograd.profiler._is_profiler_enabled` 为真时选用完整 autotuner 入口。诊断按实际 flag、进入的调用点和 C++ 绑定成功返回分别计数。[PR](https://gitcode.com/Ascend/pytorch/merge_requests/42634) · [功能边界](https://gitcode.com/Ascend/pytorch/blob/2701f7b3e8d89625752edbde10e42c2b2172272c/torch_npu/_inductor/docs/npu_inductor_fast_launch_zh.md)

## 已有实机 trace

原 01 动态 eager 的两个完整 Level1 CPU 步分别有以下提交。内部 ACL 提交按同一 consumer tid、对应 dequeue 区间内的 `aclrtLaunchKernelWithHostArgs` 调用起点统计。

|步及实际形状|全部 enqueue|其中 ACLNN enqueue|全步 ACL kernel launch|前向 embedding 请求 / 内部 launch|反向 embedding 请求 / 内部 launch|
|---|---:|---:|---:|---:|---:|
|4 · [1,19]|9,927|9,892|15,093|1,027 / 2,051|1,027 / 6,159|
|5 · [2,14]|9,923|9,893|15,094|1,027 / 2,051|1,027 / 6,159|

主线程、worker 与设备轨道均未出现可识别的 Inductor/Triton wrapper 标记。训练和模型源码通过标准 eager 的 Embedding、ATen 和 ZeRO 接口调用。E01 的训练记录还显示 8 步有 8 次 grouped 侧表命中。实现每步经一次 `ordered_64[grid](...)` 直接调用 Triton JIT。这两条路径的源码均无 !42634 的 Python wrapper 调用点。旧台账中的图回放、D01 的 CANN executor 缓存及直接 Triton fold 的执行入口也各有区别。[逐步复算](stock-level1.json) · [trace 统计脚本](inspect_existing_trace.py) · [原模型](../../evidence/baseline-steady/workspace/model.py) · [E01 入口](../../dynamic-free70/working/evidence/grouped-side-v1-l1-ab/workspace/side_tables_op.py) · [E01 激活记录](../../dynamic-free70/working/evidence/grouped-side-v1-l1-ab/02-B/rank000/grouped_side_counts.json)

## 训练开关诊断

E01 的原始 Level1 在 #4/#5 的物理流 47 各有一次 `ordered_64`。设备核分别为 71.16、71.64 µs。此前所有物理流的工作空档为 197.04、160.20 µs，全部落在官方 Free 内。可见同名 Host scope 开始前已过去 171.60、136.39 µs，此区间还包含从前一项工作到 Python 启动 fold 的代码。两步的 enqueue 分别为 10.82、10.99 µs，队列等待 2.57、2.87 µs，consumer 服务 18.51、16.80 µs，服务与设备空档的交集只有 11.38、9.40 µs。consumer 内部的 CANN `Node@launch` 分别为 8.95、7.28 µs。该 trace 未提供细分的底层 ACL launch 界限。一次/步的频次和这段空档可用于评估直接 Triton eager 适配器，单次 Host 耗时需另加局部打点。[Level1 逐段与流](e01-fold-timing.json) · [收据 SHA-256 与原文件路径](../../dynamic-free70/working/trace-index.json) · [复算脚本](fold_timing.py)

另一路 direct-publish-v1 B 的 Level0 在 #4–6 每步也是相同流上的一次 fold，实际设备时间 71.38、71.40、143.20 µs，核前空档 215.08、175.44、186.74 µs。它共用的侧表与 ZeRO 改动更多，应按该路径自己的同轮 A/B 解释。[Level0 明细](direct-fold-timing.json)

在隔离的 runner workspace 执行 `python stage.py <workspace>`。A/B 使用同一份含该 PR 的 TorchNPU 构建，进程启动前将 `TORCHINDUCTOR_NPU_FAST_LAUNCH` 分别设为 0、1。训练输出的 `rank000/pr42634_receipt.json` 记录每步实际形状、开关、扩展文件、两个 pybind 符号是否存在、自然加载的 Inductor 配置及实际计划创建和提交计数。探针仅在 Python 请求真实 bind 模块时挂接 wrapper 计数。原训练保持 eager，两腿共用 01、Mode 1，以及同一 60 步序列。

如果两腿均已确认加载包含两个符号的库，并且 `planned_submit_ok`、`bound_call` 均为零，则该开关在此 workload 未激活。此时步时、Free 或 ACLNN 的差异不构成该 PR 的实测收益。JSON 的次数映射仅保留非零键。若库符号缺失，应先标明库版本，不能用该次零计数判定工作负载适用性。

## 实际生成的动态 wrapper 正控制

独立控制算子使用 `torch.compile(fullgraph=True, dynamic=True, backend="inductor")` 对 BF16 点算子 `x * 2 + y` 生成 Python wrapper。用小整数精确检查四种实际形状的输出。每腿独立启动进程和新的编译缓存。以下命令在共享锁内、激活原训练容器环境后执行，`PR_PACKAGE` 指向对应构建，`CONTROL_DIR` 指向全新的隔离目录。

```sh
for flag in 0 1; do
  env TORCHINDUCTOR_NPU_FAST_LAUNCH="$flag" \
      INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE=0 \
      TORCHINDUCTOR_CACHE_DIR="$CONTROL_DIR/cache-$flag" \
      PYTHONPATH="$PR_PACKAGE:${PYTHONPATH:-}" \
      python compiled_control.py --output "$CONTROL_DIR/flag-$flag/result.json" --profile level0
done
```

结果保留生成的 wrapper 路径与 SHA-256、绑定行、实际 `make_plan_ok`、`planned_submit_ok` 和原入口的回退原因。先在无 profiler 的重复窗口和退出 profiler 后核对 B 的真实提交，再逐步读取 Level0 的 `torch_profiler_flag`、`original_profiler` 和 `planned_submit_ok`。另开空缓存并改为 `--profile level1` 或 `--profile torch-cpu` 可检查另一 profile 实现。该正控制只验证 PR 的编译入口和路由，原 HSTU 的收益需要在其自身入队和设备路径上验证。诊断插桩只用于激活检查。
