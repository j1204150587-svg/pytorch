# IPG 连续复制的目标视图准备与梯度绑定

在采用整桶复制的现有路径上，Level1 第 4、5 步的物理流 44 各只有一个 TensorMove，其前仍有 18.737/18.793 ms 全流空档。主线程为每个梯度创建一个目标切片，两个步骤分别有 1,053 个 narrow、1,052 个 view。所有可见 ATen 范围并集为 7.516/7.648 ms。复制后到流 47 的 ZerosLike 还有 10.879/11.100 ms，期间出现 1,052 次浅拷贝类型兼容性判断。[原始实测与逐笔关联](../ipg_range_copy/paired/RESULTS.md)

每个非空完整 IPG 桶在原布局、参数绑定、所有权、归约流、BF16 及 DeepSpeed 源码守卫通过后，A 运行已测的 Python 代码。B 调用同进程 libtorch/THPVariable C++ 扩展 `Prepared::make`，为每个真实梯度从新分配的连续目标以实际形状、连续 stride 和元素 offset 创建一个 as_strided 视图，保存在当前桶的 C++ 对象内。两腿执行同一条 Python `dst.copy_(source)`、相同两侧 record_stream、相同的目标版本补齐及优化器缓冲发布。B 随后在持有 GIL 的 `Prepared::publish` 内按原顺序逐个置归约 flag，并逐个对原始梯度执行 `grad.set_data(view.variable_data())`。该 C++ API 对应 Python `grad.data = view.data`。已有梯度对象保持原身份和自己的版本。下一桶重新检查与创建视图，计划仅限单次发布。

两腿加载同一份经 SHA-256 核验的扩展。原有缓存、分块 overflow、ZeRO 分区与 norm 设置相同，只切换 `NPU_IPG_HOST_VIEWS=python|native`。既有整桶回退保持原实现。自定义 ATen/torch function mode、compile、inference、非普通 Tensor、非标准归约 flag list、名称、子类、非连续、前向 AD 或不匹配范围，均在设备复制前弃用新视图路径。临时目标视图没有写入源或目标。完成设备提交后的异常上抛。

本地 CPU 实编门逐项核对 5 个连续优化步中的全部梯度位值、权重、入口和 post-accumulate 参数 hook、非有限跳步、原梯度对象、已有 detached alias、梯度/目标版本、视图 offset/stride、状态位及下一步。另测 5 个与 1,052 个梯度的完整拷贝和负例、模式/标志列表回退，以及尾段长度不足时零发布。[本地 CPU 门](local-cpu.json)

容器内将 R 指向 `/artifacts/dynamic-infra-20261127/dynamic-free70`，宿主的对应目录在 `/data/recsys-gr-ds019-host88/artifacts/` 下。容器操作和训练由同一宿主共享锁串行调度。

```sh
python3 "$R/ipg_native_views/build.py" "$R/ipg-native-views-build-v1"
python3 "$R/ipg_native_views/micro.py" "$R/ipg-native-views-build-v1" "$R/ipg-native-views-cpu.json"
python3 "$R/ipg_native_views/micro.py" "$R/ipg-native-views-build-v1" "$R/ipg-native-views-npu.json" --device npu
```

宿主将 recipe.json 安装为 runner_tools/recipes/ipg-native-views-v1.json。运行原训练 gate 和 L1、正反序 50 步与三步 L0。真实 B 每步应有一次完整 `prepared` 和 `published`，零回退；两腿确认同一 range marker、同流相同槽位一个 TensorMove、相同设备任务序列。按原前后设备端点复算局部空档和主线程 ATen 并集。若梯度、hook、别名或后续训练状态不一致，视为语义失败。若提交数或工作类型改变，单纯 Host 视图优化的归因不成立。最终效果使用独立无 profiler 与 Level0 配对。
