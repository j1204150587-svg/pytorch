# TorchNPU Storage 构造器分配进程内 ID

在 PTA `38e483ee79ec8a3286517808740a811c0f0ae66c` 的 `NPUStorageImpl::NPUStorageImpl` 中，序号 `idx` 是进程静态变量，但每次构造器对新建对象自己的 `unique_id_mutex_` 加锁。不同对象持有不同的锁，无法互斥分配共享序号。对目标 PTA 及 op-plugin 的源码搜索仅发现该构造器使用此锁。`get_unique_id` 是公开内联接口。候选保留类布局和成员，对共享原子计数器执行 `fetch_add(1, memory_order_relaxed)`。每个新对象在构造阶段取得返回的前值，仍从 0 开始。ID 表示对象标识，无跨线程发布其他 payload 的语义。[原生补丁](storage-id.patch) · [固定代码及最终库哈希](combined/receipt.json)

原训练 CPU-clock 的两次 8 秒采样分别有 2,156/2,103 个样本。逐个根据对应工作进程的 PERF_RECORD_MMAP2 记录，把 libc 和 PTA 帧映射回各自的 ELF，再检查完整调用栈。叶函数为 `pthread_mutex_lock` 的样本分别有 65/62 个，其中仅 1/2 个同时经过 Storage 构造器。后一次还有 11 个锁叶的第一个 PTA 帧是 `Repository::WriteQueue`，5 个是 `isDeviceCtxActive`，5 个是 `copy_d2d_baseformat_opapi`。16 个锁叶未还原出 PTA 帧。这些 CPU 采样有限，无法观测线程阻塞期间的时长。[逐栈复算](perf_stacks.py) · [完整样本分类](perf.json)

在无卡 `--runtime=runc --network=none` 容器中，以真实的 C10/ATen 和本次修改的完整 NPUStorageImpl 构造器代码构造零字节、CPU-owned 的存储对象。夹具保留完整构造、析构和 ID 读取，逐个固定工作线程的 CPU。关闭诊断，单线程 ABBA 运行三轮，每个独立进程创建 500,000 个对象。A 的每对象中位数为 36.828 ns，B 为 21.810 ns，差 15.018 ns。单独对 B 使用 2/4/8 个线程，各创建 200,000/400,000/800,000 个对象。每个进程所有 ID 经排序后都是从 0 开始的连续序列，未出现重复或缺失。[夹具](storage_cpu.cpp) · [编译和跑法](cpu.py) · [15 次原始结果](cpu.json)

训练端用同一份动态 eager 脚本、模型和双向 embedding wheel，A/B 保持 01、缓存 65536、任务队列模式 1 和单 rank。只切换 `TORCH_NPU_STORAGE_ATOMIC_ID`。八步审计共用 `TORCH_NPU_STORAGE_ID_AUDIT=1`，分别记录构造次数及 ID 代码段的 `steady_clock` 耗时，含计时器开销。实验库 SHA‑256 为 `9506bc90c7b629a14ed7a31e14c851bc98b0594ef2459a260283d8138f3f2862`。[构建器](build.py) · [实机审计](audit.py)

## 尝试无优化路径

原训练八步含初始化与收尾，两腿各创建 105,703 份 Storage。ID 代码段 A 累计 6.211 ms、最长 0.690 µs，B 累计 7.372 ms、最长 9.510 µs。A/B 的梯度、范数、BF16/FP32 权重和 AdamW 状态相同，loss 最大绝对差为 3.81×10⁻⁶，均完成 8 次更新。逻辑 8 映射板 4、芯片 0，两个窗口均有有效工作进程采样，未见其他占卡进程。[原计数和状态比较](audit.json) · [采样校验](slot.json) · [原始运行、命令、调度与形状](evidence.tar.gz)

按无卡构造器差值 15.018 ns 和审计实际总数估算，只对应 1.587 ms/进程，即使全部归入训练也约为 0.198 ms/步。这是局部成本的外推。设备上的诊断代码段累计时间也未改善，因此本项止于八步审计与并发 ID 验证。没有该机制的 10+50 双序端到端、L0 Free 或 L1 提交数据，不能认领训练速度或空档收益。目标构造器对进程共享的序号计数使用单一原子变量，可独立提出为并发正确性修正。
