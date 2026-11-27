# 动态 eager 分区梯度布局检查

`flat_stats.install` 安装的 overflow 与 norm 调用 `eligible(shards)`。01、07、分块 norm、#09 平坦梯度提示、侧表 slab、flat stats 和 ACLNN executor cache 在两腿配置相同。每次成功调用读取 1,052 份梯度的布局。有限训练步执行两次，检测到 overflow 的步骤只执行 overflow 检查。新开关只替换循环。

两腿先核对 `zero._recsys_local_flat_gradients[0]` 中的列表身份、长度和 flat 的设备、dtype、形状、格式、连续性与梯度状态。循环逐个核对 Tensor 类型、视图根、StorageImpl 身份、dtype、设备、requires_grad、neg/conj 位、形状、连续性和 offset。B 只缓存初始化时预期的形状与元素数，每次从本轮真实张量读取相应元数据。针对自定义上下文和无法确认的张量继续运行 A 的 Python 检查。外部替换、重排或重绑视图时，原 overflow/norm 运算仍可回退。

`cpu-micro.json` 留存旧 Python 检查在重绑 storage 后依然接受伪视图的证据。`cpu-micro-tightened.json` 记录两腿均检查真实 StorageImpl 后的 14 个用例、模式回退与本地 Windows Torch 2.9.1 的交替计时。成功路径五次 native guard 的 CPU autograd profiler 没有记录 ATen 运算。设备提交次数和空档归因由同一步、同 stream 的实际 Level1/ACL 测量确定。

在训练容器中构建扩展并运行同版本 CPU 微测。

```sh
R=/artifacts/dynamic-infra-20261127/dynamic-free70
python3 "$R/native_guard/build.py" "$R/native-guard-build"
python3 "$R/native_guard/micro.py" "$R/native-guard-build" "$R/native-flat-guard-cpu.json"
```

正式配方为 `runner_tools/recipes/native-flat-guard.json`，仅改变 `NPU_DYNAMIC_NATIVE_FLAT_GUARD`。每个进程以 `native_flat_guard.json` 记录分支选择、接受次数、回退次数，以及它们与 flat stats 实际次数的对照。Level1 区间为 `PTAFlatGuardPython` 和 `PTAFlatGuardNative`。归因须检验每步两次 Host 区间的缩短是否提前了下一次关键设备任务。若正式窗口两序的 E2E 或 Free 未保持改善，微测数据仅用于解释 Host 检查本身。
