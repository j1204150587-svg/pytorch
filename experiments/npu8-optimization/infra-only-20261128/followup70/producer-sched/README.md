# 动态 HSTU 的 producer 调度归因

沿用同一容器、同一逻辑 NPU 8、01 分块 overflow、Task Queue 1、1,024 个侧表、单 rank、梯度累积 1 和 BF16 ZeRO-2。A/A 都启用双向 embedding indices 描述符规范化与 65,536 项 CANN cache。两腿保持原始模型、动态输入生成、训练脚本和 runner。诊断只回答已经形成的全物理流空档中，后继提交对应的 producer 在进入负责这笔提交的 ATen 前是否在运行，以及稀疏的用户态栈实际落在哪些函数。结论只覆盖该轮 Level1，不混用其他 run 的 Sample 或 Free 数据。

每次先从本轮 dispatch 的唯一输出路径和 LOCAL_RANK=0 找到真实训练 PID，再通过 /proc/<pid>/task/*/status 的 NSpid 将主线程及 pt_autograd_0 的宿主 TID 映射到 torch trace TID。目标列表上的 perf record --no-inherit --switch-events --clockid mono 记录上下文换入及主动/抢占换出和 499 Hz 的用户态 cpu-clock FP 栈，与整机 sched_switch 的采集分开。第二个相同 TID 列表上的 perf 只记录 aclrtSynchronizeDeviceWithTimeoutImpl 的稀疏入口及返回探针。无卡短进程验证可捕获 IN、OUT 与 OUT preempt。首次观察器采用 Python 文本追加打开 tracefs，内核拒绝 seek；在本轮训练排队期间改为 os.open/O_APPEND，随即成功安装，并保留首次日志。[采集程序](sample.py)

同次采集按 25 ms 间隔取六组 REALTIME→MONOTONIC→MONOTONIC_RAW→REALTIME 时间，保留每组最窄夹逼。perf 明确选择 MONOTONIC。用同步函数进入与退出约束同 TID、同名、端点均足够接近且双向唯一的 CANN API，在校验覆盖处给出主机 trace 与 perf 的偏差界。设备时间的界另由同次 trace 的逐个已链接 ACL 提交与全设备同步收紧，假设同一步内时钟相对偏移固定。申报保守区间时同时考虑这两侧误差。

原始关联使用前一项全流实工作末尾、下一项全流实工作开始、唯一 HostToDevice→CANN→dequeue→enqueue 及包住 enqueue 的最内层 ATen 起点，涵盖 kernel 与数据搬运。每个空档的名义前缀为从全流空档开头到下一项工作开始或拥有该 enqueue 的 ATen 开头，取较早者。对保守结果，把左端移至设备时钟允许的最晚开始，把右端取设备最早结束与 ATen 开头的较小值，再按 host perf/trace 的误差收缩两端。任何一界无解或有观测丢失时不申报这部分。只有在相反的两条相邻 switch 记录之间，才归为在运行、抢占换出、其余换出。无边界状态为未知。其余换出涵盖主动等待等类型，无法仅凭其推出 GIL 或具体等待对象。每条用户态栈只代表该次采样瞬间，系统函数、CPython 或框架的样本不扩展到整个所在区间。[同窗复算程序](analyze.py) · [原始结果打包](collect.py)

`src/acl/aclrt_impl/device.cpp:163` 的 `aclrtSynchronizeDeviceWithTimeoutImpl` 首个 C++ 局部对象由 `ACL_PROFILING_REG(acl::AclProfType::AclrtSynchronizeDeviceWithTimeout)` 建立，函数返回前结束该记录。源文件 SHA-256 为 `dc14bbfd1cb0222b96d9c9e6115facbeb86f853189c358dbeda83bb41e912fbb`。实机 `libacl_rt_impl.so` 的 SHA-256 为 `4595d4cf50f064cc9d6a97688eecf5e95ef92a4291911e26dd301091997c37ee`，该导出符号在地址及文件偏移 `0x412d0`。入口、返回夹住这个 CANN 记录，逐次约束 perf 时钟的上、下界。
## 同窗结果

两次原始 Level1 各覆盖第 4 步 `[1,19]` 和第 5 步 `[2,14]`。完整 60 项序列为 B=1–4、L=12–32 的合成形状。时钟桥在两个进程分别记录 998/981 组四时间戳。常量 REALTIME−MONOTONIC 映射的上下界无法相交。同窗实测 RAW−MONOTONIC 在约 25 秒内单调变化 −1.002/−0.984 ms，与 ACL 记录约 −40 µs/s 的漂移一致。按两相邻采样的 RAW−MONOTONIC 线性插值转换每条 perf 时间，用实测相邻最大差及读时钟跨度保留误差；各 5 组独立同步探针约束均相交，得到 3.856/3.680 µs 的保守主机误差。四步的设备偏移区间分别为 [−1.405,+226.805]、[−17.495,+155.505]、[−6.070,+152.310]、[−22.660,+143.410] µs。两套 perf 数据均无 LOST 记录。训练任务退出后计时用 sleep 提前收到终止信号，第一版脚本把 perf 的 −15 误判为不能解码；后续直接离线解码保存的同次 perf.data，核对了探针、时钟夹逼和完整 Step 窗口。整芯片 8/7 次采样各看到目标 PID 4 次，无外部进程。[A 逐端点](01-A-sched.json) · [B 逐端点](02-B-sched.json) · [物理卡收据](slot.json)

|腿、步|实际 shape|名义后继 ATen 前缀 ms|保守前缀内 on-CPU ms|保守前缀内被抢占 ms|保守前缀内其他换出 ms|
|---|---|---:|---:|---:|---:|
|A，4|[1,19]|96.996|37.626|0.182|2.241|
|A，5|[2,14]|79.133|39.245|0.000|3.366|
|B，4|[1,19]|128.871|40.744|0.000|1.964|
|B，5|[2,14]|99.203|38.028|0.000|1.440|

四个窗口中，最长的保守前缀都在物理流 47 的后继 `aclnnMul` 之前，producer 为 autograd 线程。A 第 4/5 步的 queue ID 为 51353/61275，保守前缀 13.755/13.306 ms；B 对应相同两个 ID，18.103/16.610 ms。不能将其解释为 `aclnnMul` 的执行或下发延迟。

所有保守前缀覆盖 75 个实际 on-CPU 样本，24 个栈顶位于 CPython、14 个位于 PyTorch CPU、12 个位于 PyTorch Python 绑定、20 个位于 libc/allocator/系统运行库、4 个位于 NPU 软件栈，1 个位于 profiler。其中能解析的具体栈顶含 `_PyEval_EvalFrameDefault`、`_PyType_Lookup`、`torch::autograd::deleteNode` 和 `c10::TensorImpl::grad()`。这些采样瞬间确认了设备空档中有主机侧解释器和 autograd 活动；CPython 内的具体业务 Python 行、真实阻塞对象、未覆盖空档，以及样本间 CPU 时间归属尚不确定。两腿均使用相同实现，以上归因没有性能 A/B 或单次下发收益。完整包包含原始双腿 perf、校时点、Level1 trace、实际 shape、scope/queue/stream 精确关联、逐空档状态与逐样本栈。[完整原件](raw-evidence.tar.gz) · [按 ELF 代码段重符号](resolve.py)
