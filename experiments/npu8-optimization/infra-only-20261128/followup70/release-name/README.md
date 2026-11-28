# PTA release 环的 OPAPI 名称复制

PTA `38e483ee79ec8a3286517808740a811c0f0ae66c` 的 `OpParamMaker.cpp:713` 在 release 环传递 `EXECUTE_OPAPI` 时调用 `ExecuteParasOpApi::Copy(ExecuteParasOpApi&)`。目标 wheel 的分支在 `libtorch_npu.so+0x3c5d19c` 调用该方法；`+0x3c4184c` 装载 99，`+0x3c41874` 调用 libc `strncpy`。即使名称很短，`strncpy` 也用零填充到 99 B。随后 move 同一个 `std::function<int()>` 至 release 槽，清空源 handler。release 线程的 `ReleaseParamFunc` 对这个类型只清空目的 handler；释放、报错、日志均不读取 release 副本的 opType。原执行队列仍需自己的名字去做报错和 profiling。`COMPILE_AND_EXECUTE` 的 release 交接只调用 CopyEx 搬 ACL 对象，事件等其他队列项不进这个 strncpy。[目标 wheel 指令与 SHA](instructions.txt)

原级别 1 的完整 queue ID 记录中，第 4/5 步有 9,892/9,893 条 aclnn 命名请求，在 [1,19]/[2,14] 各首次出现。另有 36/30 条事件或 HcclAllreduce。名称分布用于构造微测负载，不能单凭拼写确定原生 paramType。[真实顺序](request-names.json) · [完整 Level1 请求及空档](../lazy-gil/linked-level1.tar.gz)。两份原始 CPU profiler 中，各有 6/7 个样本栈进入 `CopyReleaseParamFunc`，占全 rank 2,156/2,103 个样本。两腿各 1 个样本落在直接受 strncpy 调用的 libc 内，其余 5/6 个主要落在 std::function 的 move 指令 `+0x3c41894` 等位置。[各栈与地址](perf.json) · [解析程序](perf.py)

无卡容器使用相同训练镜像和同一份 PTA 编译数据库，直接构建固定源码的 `NPUDefine.cpp`。A 调用原方法，B 仅调用单独编译的同签名 move，源的 `Release()` 在两腿保留。按原名字顺序、真实 544 B 槽距和 136 B payload 布置，两腿预先对该类型正常构造，保证 C++ 生命周期有效。310 核逐批交接，311 核实际调用 handler 并清理，校验源已移空以及堆捕获引用计数复原。每种场景先用 4 批预热，再执行 16 组 ABBA，共 64 批，A、B 各 32 批。[微测代码](release_cpu.cpp) · [B 的 move](move_only.cpp) · [构建与可执行文件指纹](cpu-receipt.json)

|槽数|捕获对象|A ns/项|B ns/项|差值 ns/项|
|---:|---|---:|---:|---:|
|64|小对象|12.188|7.266|4.922|
|64|持有共享对象的堆捕获|13.281|8.750|4.531|
|8,192|小对象|14.962|5.981|8.981|
|8,192|持有共享对象的堆捕获|15.845|7.616|8.229|

每种配置在 16 个 ABBA 分组里均同向。用 8,192 槽的分解差值乘上述约 9,900 个调用，得到每步 0.081–0.089 ms 的 CPU 预算，尚未计入与设备计算重叠或等候后继 producer 的时间。该值用于判断实机训练的收益量级。[原始 CPU 窗口](cpu.csv) · [配对差、波动与预算](cpu-summary.json)

目标 release 环通过 `NewFunc` 一次 malloc 并 memset，常规执行队列的 `CopyFunc` 会 placement-new 目的对象，release 槽的 `CopyReleaseParamFunc` 没有对应的显式构造。若推广该改动，需要为 release 槽的 handler 显式建立并结束生命周期，保持执行队列中原名可用于错误路径，同时审计同一个槽在 OPAPI、COMPILE_AND_EXECUTE、事件之间复用及退出排空。直接对零填充的非平凡 `std::function` 做 move assignment 不足以验证正确性。考虑原生采样与微测预算，该方向先止于独立成本分解，不认领单次 ACL、Free 或端到端收益。
## 原训练中的分支计数

原始八步训练两次均在 PTA 加载前将三条针对实际 rank 的 shipping uprobe 加入 perf。每个进程记录到 `CopyReleaseParamFunc` 87,845 次，其中 OPAPI 名称复制 80,233 次（91.335%），COMPILE 分支 0 次，剩下 7,612 次走其他分支。这个完整进程的计数包含初始化和收尾。两个 rank 的事件计数三项完全一致。首次监视器遇到 `/proc/<perf pid>/fd` 瞬时消失并提前撤掉探针，首次八步不用于命中率；已在后续独立 A/A 修复该竞态并验证 attach。[分支计数、卡锁及指标](activation.json) · [原始 CSV、命令与训练数据](evidence.tar.gz)

实际 A/A 八个不同 shape 的梯度、norm、权重及 AdamW 状态一致，各完成八次有限更新，无自然 overflow 跳步，未注入非有限值。loss 最大绝对差 3.8147×10⁻⁶，通过原 runner 的容差门。整芯片两窗分别采样 6/5 次，均捕获对应 worker，未发现外部进程。CPU 微测的 0.081–0.089 ms/步是同类型工作缩短的预算；这项改动没有实测训练提速或 Free 下降。
