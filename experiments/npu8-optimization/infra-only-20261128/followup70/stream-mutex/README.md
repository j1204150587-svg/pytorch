# CANN v100 单次提交末尾的 StreamMutex

在相同的动态 eager baseline 的两组 8 秒、199 Hz 用户态 CPU 采样中，`acl_thread` 的 `libc.so.6+0x84ea4` 栈顶分别有 63 和 55 个样本，其直接上一帧是 `libruntime_v100.so+0x2f67dc` 的有 59 和 52 个。整 rank 分母为 2,156/2,103。该处是 `AllocTaskAndSendStars` 调用 `StreamUnLock` 后的返回点；libc 的指令在交换 mutex 锁字后检查旧值，必要时才走 futex 唤醒。两组 on-CPU 样本均不提供旧锁值、等待时间、内存地址或 cache 命中来源。[perf 重符号](stock_stacks.py) · [两组计数和原件 SHA-256](stock-stacks.json) · [原始 perf](../host-samples) · [shipping ELF 指令](instructions.txt)

公开 CANN v9.1.0 提交 `0b4af8a31c75a1f1f43638a8f75b9e3c6182e031` 中，`src/runtime/core/src/task/task_submit/v100/task_submit.cc` 第 423/459 行获取锁，正常路径第 588 行在 HAL 提交之后释放。持锁期间的队列尾、SQE 构造、等待、HAL 提交及错误分支影响同一 stream 的提交顺序。`stream.hpp` 第 1532–1533 行依次声明两把 `std::mutex`；`stream.cc` 第 2519–2551 行实现二者的 lock/unlock。安装版本的锁字分别在对象偏移 +0x28 和 +0x58，距离 48 B，实际共行取决于该次分配的基址。第 2650、2681、5098 和 5216 行还通过 guard 直接取得同一个 streamMutex。

若多个线程修改同一活跃实例的两把锁，且两者在目标层级实际共行，才继续判断伪共享。先核对两把锁的地址、写入线程与 CPU，以及相同锁的等待路径。若修改者相同、两者落在不同行，或第二把锁在诊断窗口不活跃，便不把这段释放的 CPU 样本归因为两锁伪共享。隔离锁字段会改变 C++ 布局和公开成员偏移，只有在访问路径与 ABI 均能覆盖时才有可验证的补丁。提前解锁还须维持同流排队与 HAL 提交的原顺序。

专项诊断复用原共享锁、原训练 runner 及双向 embedding 描述符规范化、CANN cache 65536、工作量和开关相同的 A/A 配方。每次仅对本轮 rank 开启 0.65 秒的六个 shipping v100 uprobe 及 futex 系统调用，采集调用前锁字、成功获取、释放前锁字、对象地址、线程和 CPU。随后关闭这些事件，再在同一进程采 0.7 秒 user-mode ARM SPE，以当次对象的地址筛选实际读写、延迟和缓存事件。探针会增加调用的墙钟时长；它提供线程、锁字和相对时间信息，不能作为未插桩的单次发布时延。SPE 的逐包采样会漏掉实际访存。两个窗口均仅作归因。[诊断驱动](sample.py) · [复算](analyze.py)

两个独立进程里最频繁提交的对象基址均为 96 mod 128。其 streamMutex 与 recycleMutex 的锁字分别落在同一 64B 行内的 +8 和 +56。`acl_thread` 对 streamMutex 分别记录到 18,537、26,201 次获取，`RT_RECYCLE_0` 对该对象的 recycleMutex 记录到 512、488 次。两把锁在获取前均观察到锁字 0，释放前均观察到 1。过程内 996/1,089 次 futex 进入均未匹配这两个最热的锁地址。独立的 0.45 秒 glibc 入口探针按该进程所有目标锁地址过滤，可覆盖直接通过 guard 的调用，已捕获的 streamMutex 锁字同样为获取前 0、释放前 1。宿主 L1D/L2 行宽 64 B，L3 为 128 B。SPE 两窗解出 353,557/327,716 个本 rank 用户态访存包，在覆盖各 Stream 两锁的 128B 行上匹配 551/594 个。该 PMU 数据中最热对象的两侧出现两个线程的处于锁字段范围内的 load；写入和线程归属另由确定的 glibc normal mutex 实现、进入前和释放前锁字以及调用线程确认。采集日志有 ring/AUX 丢块，报告的调用数与包数均是已记录样本；没有同一行的硬件所有权迁移次数，也未逐项对齐某个设备空档。[原始诊断与命令](evidence.tar.gz) · [逐对象观测](analysis.json) · [复算摘录](measurement-summary.json) · [全调用方的地址过滤](all_callers.py)

按时间搜索两次探针窗口最热对象的连续提交，在已能配对的上次 stream 解锁至下一次 stream 取锁之间，实际看到另一个 CPU 对 recycleMutex 解锁的为 264/18,528（1.42%）和 243/26,188（0.93%）。回收也可能发生在提交临界区内。这个直接交错统计与硬件缓存行所有权迁移次数各自有不同的观测边界。[时间排序程序](handoff.py) · [逐进程计数](handoff.json)

在原镜像的无卡容器中按这两个进程的主要 CPU 对 188/189、266/269 及最热对象提交与回收次数比 36:1、54:1，使用相同的 glibc 普通互斥锁。每组强制回收写完成后再执行一批提交线程的锁/解锁，两种基址按交替正反序重复，各计 6 个进程内窗口、每窗 8,500 个有效批次。从原观测位置 96 mod 128 切到 64 mod 128 后，两核对受交接影响的第一笔 P50 为 70→40 和 130→40 ns，36 次批 P50 为 900→870 ns，54 次批为 1,380→1,290 ns。折算到这个强制交接微测的每笔锁操作为 0.83/1.67 ns，限于其固定比率及核心亲和性。[测试代码](placement_cpu.cpp) · [逐窗口数据](placement-cpu.csv)

实机候选只调整 `src/runtime/core/src/stream/v100/stream_creator_c.cc::CreateStreamAndGet` 的普通 Stream 分配。A 走原 `new (std::nothrow) Stream`。B 从最多 12 块普通全局 operator-new 原始存储中选择 streamMutex 活跃字段末尾与 recycleMutex 锁字跨越 64B 和 128B 边界的一块；找不到时回退首块。其余块由 RAII 原样释放，在选中处构造同一精确类型，原虚析构仍对同一全局 allocator 返回的指针执行 delete。只在创建时选择，已有类布局、持锁操作及 HAL 提交次序都保留。该原型可能增加 Stream 创建的暂时堆用量，计数器分别保留创建、成功分离、请求、原始分配数和回退。私有库 SHA-256 为 `1cd3aa9233a36dbd9e0109f9219451712656ef48e03705d826bb334d5e52af97`，只重编 v100 Stream factory，链接其余 pristine 对象。[原生补丁](stream-placement.patch) · [构建回执](build-receipt.json)

第一轮微门的观测器将 ACL 句柄当成对象地址，运行在开始计时前因无法核对这两个地址而退出。`src/runtime/api/api_c_stream.cc::rtStreamCreateWithFlags` 返回 `ExportEmbeddedHandle(realStream)`，其值是对象的 `GetInnerHandle()`。修正后的观测器在 factory 记录这两种地址，每条测量腿都新建同一创建顺序的普通 ACL stream，热循环前同步框架已生成的实际尺寸输入。若两序的真实 ACLNN 延迟没有稳定改善，或 A/B 的目标 stream 几何并无差别，分别标为未复现或缺少激活差，不将 CPU 强制交接微测计作训练收益。[第一次门结果](observer-first-summary.json) · [实机微门](micro.py)
## 原生实卡微门结果

逻辑 NPU 8 上四个独立进程按 AB、BA 运行，每腿 60 个实际 BF16 shape 的元素结果均逐比特相同。两次 A 的受测 stream 基址为 80/112 mod 128，两次 B 均为 48。A 的 8 条普通流中有 1/3 条自然分离；B 均 8/8 条分离，分别使用 47/43 块原始存储，均未回退。四腿按共享锁串行，四份 PID 收据可查。整芯片 7 个周期快照覆盖前三个短进程，末腿在采样间隔内结束，已采样记录中无外部进程。[分配与逐窗结果](micro-results.json) · [卡采样覆盖](micro-slot.json)

按每次真实 `aclnnAdd` 折算，全 60 shape 的 host 窗口中位数在 AB 为 3.7349→3.7843 µs，在 BA 为 A 3.8925、B 3.9312 µs。B−A 为 +0.0494/+0.0387 µs，窗口末尾同步后为 +0.0549/+0.0197 µs。逐笔 p50 的 B−A 为 +0.021/+0.109 µs，p95 为 −0.061/+0.609 µs。第 4 个 shape 的 host 窗口差为 +0.1236/−0.1772 µs，第 5 个为 −0.0947/−0.0734 µs，后者逐笔 p50 仍慢 +0.140/+0.110 µs。JSON 中 `host_ns` 与 `complete_ns` 是源窗口名；汇总后的单位为 µs/次。两个实际 shape 的 A/B 四个 Level1 区间各有 32 个 ACL 和 32 个 `aclnnAdd_AddAiCore_Add`，均在物理流 45。完整形状范围的提交延迟在两种次序下均未改善，在原生微门结束该候选，无 HSTU 训练时长或 Free 收益数据。[原始 CSV、trace 与命令](micro-evidence.tar.gz) · [通用真实 ACLNN 测量代码](../cann-pending/launch_probe.cc)
