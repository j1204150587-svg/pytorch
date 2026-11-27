# 动态 eager 侧表参数检查

`SideSlab.forward` 每次进入设备端侧表求和前扫描 1,024 张表。可切换的 C++ 方法在一次 CPython 调用中遍历当步的 `nn.ModuleList`，逐张核对 module、weight、data_ptr、requires_grad、grad、4 种 module hooks 和 2 种 parameter hooks。它通过标准属性访问读取实时值并按原顺序短路。自定义表容器或非标准快照走保留的 Python 谓词。模型模式、全局 hook、ZeRO flat 及输入检查仍在调用点的外层执行。

目标函数每步一次。A 的 `PTASlabParameterGuardPython` 和 B 的 `PTASlabParameterGuardNative` 用于核对 Level1 关键路径。成功路径不更改侧表运算、梯度发布或设备任务。若观察到同一步的设备任务数量/stream 改变，应重新核对归因。若减少的 Host 时间被先前已入队的设备工作覆盖，则该变化不能计为关键路径收益。

CPU 微测 `cpu-micro.json` 保留 17 个正负样例的 A/B 判定、两次顺序交替的 60 对计时和每腿五次扫描的 ATen 事件清单。正式验证应使用训练容器配方 `runner_tools/recipes/native-slab-guard.json`。两腿共用 01、07、分块 norm、缓存、#09、分组侧表和 slab。

```sh
R=/artifacts/dynamic-infra-20261127/dynamic-free70
python3 "$R/slab_host_guard/build.py" "$R/slab-guard-build"
python3 "$R/slab_host_guard/micro.py" "$R/slab-guard-build" "$R/slab-guard-host-cpu.json"
```

每次训练在原 `side_slab_counts.json` 内记录 native、Python、接受、拒绝和不支持容器的回退计数。`guard_accepted` 和 `slab_forwards` 应等于步数。
