# TorchNPU 已完成 Python 初始化时跳过重复 GIL 获取

PTA `38e483ee79ec8a3286517808740a811c0f0ae66c` 的 `NPUNativeFunctions::empty` 通过 `maybe_initialize_npu` 调用 `torch_npu::utils::npu_lazy_init`。目标 wheel 的 TensorFactories 对象保留了对该函数的真实符号引用。原版在每次调用构造 `pybind11::gil_scoped_acquire`，再读由 GIL 保护的 `npu_run_yet`。模型不变、双向 embedding 和缓存 65536 都开启时，#4 [1,19] 和 #5 [2,14] 各有 7,743/7,744 个 `empty_tensor`。该标记独占 CPU 24.32/23.82 ms，p95 为 5.33/5.22 µs，p99 为 9.01/7.34 µs，最大 136.71/201.98 µs。与后继请求尚未入队的设备空档相交 8.910/5.863 ms，其中还包含分配器、stride 及存储描述符准备。既有 8 秒 CPU 采样在 B 腿主训练线程观察到 `npu_lazy_init` 调用链，仅能定位在 CPU 上执行的样本。GIL 的等待时间另由原生诊断测量。[交集与分位数](../native-host/exposure.json) · [perf 原始数据](../host-samples/02-B.data)

候选只在 Python 运行中、未进入 finalization 且初始化结果已发布时，直接从原生入口返回。冷路径沿用原 GIL、异常传播、Python `_lazy_init` 和其延迟回调。只有 Python 的 `_initialized` 为真且 `_is_internal_in_bad_fork` 为假才以 release 发布，避免同线程重入时提前发布。热路径以 acquire 读取。目标源码中唯一清除 `npu_run_yet` 的调用者是 multiprocessing after-fork hook，候选在同一钩子中先清除发布标志。终结期保留原行为。未改变其他设备初始化、NPU stream、队列或 ACL launch。若实机计数没有降下 GIL 获取次数，或微测出现异常/死锁，即中止。若配对计时和 Free 证据异向，分开记录两者。[原生补丁](lazy-gil.patch) · [初始化实现](install.py) · [发布及诊断](config.inc)

测试库 SHA‑256 为 `d55a914ae13aa7703a227adba5b3fb12c161a81127e25c24875f1a9b606de68a`。使用无卡 runc 容器独立构建，链接已验证双向 embedding 的相同三个对象，未链接 view、empty 或 release-ring 实验对象。A/B 唯一机制开关为 `TORCH_NPU_LAZY_GIL_FAST=0/1`，诊断在两腿均设 `TORCH_NPU_LAZY_GIL_AUDIT=1`。计时与 Level0/Level1 两腿均为 0。[原对象与编译命令](v1/build-receipt.json) · [最终链接](combined/receipt.json) · [串行运行程序](series.py)

独立 CPU 生命周期测试以真实 CPython 初始化、注册 C 终结回调、终止和重新启动两轮。候选谓词在两次启动后都为真，在首次启动之前、两个终结回调及每次终止后都为假。[测试](lifecycle.cpp) · [回执](lifecycle.json)

两腿实机分别从冷进程启动，覆盖首次延迟回调故障后重试成功、两个并发线程等待一次初始化、初始化回调内重入、multiprocessing fork 子进程连续两次拒绝 NPU 重初始化且父进程仍可用、成功初始化后两个工作线程并发分配。延迟回调第一次抛错时，两腿均经过目标 wheel 既有的 `NameError`（未定义 `DeferredNpuCallError`），异常链包含最初的测试错误。第二次回调成功。并发分配进程的 129 次检查，A 获取 GIL 129 次，B 获取 1 次。[微测程序](micro.py) · [A 结果](micro-0/micro.json) · [B 结果](micro-1/micro.json)

在八步真实训练进程中，两腿各调用 `npu_lazy_init` 71,592 次。A 全部进入 GIL，实测范围从构造 `gil_scoped_acquire` 前到构造完成后，合计 17.942 ms，最长 10.760 µs。B 首次获取 1 次耗时 1.680 µs，余下 71,591 次从热路径返回。这些是进程总数，包含初始化与收尾。测量不含 GIL 析构，且未与 Level1 每个设备空档逐笔配对。两腿真实训练的 loss 最大绝对差为 3.81×10⁻⁶，原八步门通过，梯度和范数、BF16/FP32 权重和 AdamW 状态相同，均更新八次，无自然 overflow 跳步。[训练语义与计数原始回执](audit.json)

独立的 CPython embedding 程序检验了两轮解释器初始化、进入 finalizing 和退出的状态谓词。整套 TorchNPU 在同一进程 `Py_Finalize` 后重新载入的行为尚未验证，原有 `npu_run_yet` 的显式复位走 after-fork 路径。

同一实际工作下，两步 `empty_tensor` 的独占 CPU 分别为 24.172→22.209、24.089→23.508 ms，入队前空档交集为 8.640→6.869、6.257→5.661 ms。两步逐流设备工作逐位一致（17,220 和 17,221 项），第 4 步 B 的 Host 多两条 destroy_event 入队。单次 ACL kernel launch p95 为 7.77→8.25 和 7.90→8.31 µs。两轮 Level0 的 Free 占比均下降，分别相对改善 0.434% 和 0.920%。无 profiler 的端到端一轮快 6.035%，另一轮慢 5.265%，配对差的样本标准差为 31.376 ms/步，单独一轮的整步差值不能归给 GIL 获取。本项记录在「尝试无优化路径」，保留 Free 与 producer 端的局部观察。[八列、吞吐与单次分位](实机结果.md) · [逐笔对应](linked-level1.tar.gz)
