# CANN v100 单次提交的 stream pending 计数

双向 embedding 动态 eager 的两次 Host perf 采样中，`acl_thread` 的 `libruntime_v100.so+0x2f5d3c` 分别占全部用户态采样的 3.29% 和 3.38%。采样使用 `cpu-clock:u`、199 Hz、每次 8 秒，覆盖该 rank 的所有线程。shipping ELF 的这条指令是 `ldr x21,[x19,#304]`，紧邻它之前的四条指令是对 stream +0xf4c 的 32 位独占读写循环。公开 v9.1.0 的 `AllocTaskAndSendStars` 在 `src/runtime/core/src/task/task_submit/v100/task_submit.cc:502` 执行 `stm->pendingNum_.Add(1U)`，随后访问 model。源文件名、报错行和反汇编的调用次序一致。perf 未精确计数独占写失败、缓存线迁移、后继 load 停顿或实际经过该处的请求，采样比例也不能换算为训练收益。[采样 A](../host-samples/01-A.report) · [采样 B](../host-samples/02-B.report) · [指令核对](instructions.txt)

候选仅试该次递增。源码的 `Atomic<uint32_t>::Add` 调用默认 `seq_cst fetch_add`。A 在这处显式使用与安装版相同语义的 `ldaxr/add/stlxr/cbnz`；B 使用返回旧值到普通寄存器的 `ldaddal`，保留 acquire/release 顺序。私有库根据 Linux `AT_HWCAP & HWCAP_ATOMICS` 启用 B，逻辑流和其他计数器均沿原 CANN 实现。对照的两腿载入同一源码编译产物，分派及关闭的计数分支相同；源码构建的整个代码布局不同于安装版。公开 CANN v9.1.0 提交为 `0b4af8a31c75a1f1f43638a8f75b9e3c6182e031`。安装版 SHA256 `ab4b573989e6b76e129204cf329bdd916739bdd16092a0efcfa88d8ba011d689`，候选 SHA256 `7184371068a895f980c85251ce109b9aee485e83980b8ecde0d824c92a141e91`。在原训练容器的依赖环境下 `ldd -r` 未报告缺失符号。[增量补丁](candidate.patch) · [编译、重链与 SHA](build-receipt.json) · [源码构建脚本](build.py)

用无设备构建容器的八个 CPU 线程并发测试相同原子对象，每腿期望 1,600,000 次。实机微门使用同一 NPU 8、原 admission 和 NPU 卡锁、双向 embedding overlay、CANN cache65536 和 Mode 1。先验证安装版和私有 A/B 的 60 个真实尺寸的 BF16 加法，核对实际 worker 的唯一 runtime 映射和原生分支计数。每个尺寸独立预热正式 CANN repeatable `aclnnAdd` executor，提交时直接调用 stage 2，记录 Host 总时长、首尾同步窗口、单笔 P50/P95/P99 和长尾；加法不取代训练中任何 eager 边界。这套独立测量只用于隔离 v100 单次命令开销。[并发检验](atomic_test.cc) · [真实 launch 驱动](launch_probe.cc) · [调度与原始窗口](micro.py)

无计数的独立进程按 A→B、B→A 两种顺序运行，对 [1,19]、[2,14] 每窗 512 次，60 shape 按原序轮换每窗 480 次，各 21 窗。额外在同进程同流、计数关闭时交替 12 次 A→B 和 12 次 B→A，切换在同步窗口以外。调用级别的 Level1 四个标记各执行 32 次，两种模式核对同物理流、同 kernel 和实际 ACL launch 次数。若未观察到双序及同进程可重复的单次 launch 改善，不将采样热点记作新收益。

## 实机单次提交

安装版、私有 A、私有 B 的 60 个尺寸均逐位通过，三类进程各自只映射预期的 v100 运行库。诊断窗口 240 次真实 stage 2 对应 A 计数 [240,0]、B 计数 [0,240]。CPU 并发检验的两种原子各得到 1,600,000。四个 Level1 标记均为同一物理流 47 的 32 个 `aclnnAdd_AddAiCore_Add` 和 32 次真实 ACL launch；shape #4 的 ACL P50 为 A 3.820、B 4.065 µs，#5 为 A 4.265、B 4.125 µs。[CPU A](atomic-A.json) · [CPU B](atomic-B.json) · [原始 CSV、trace、激活与数值](micro-evidence.tgz)

下表左两列对比四个先后独立启动的进程，右两列在同一进程、同一物理流、同一批地址上只切换原子指令。正数为回退，单位 µs/launch。

| 实际 shape | 独立进程 AB 的 B−A | 独立进程 BA 的 B−A | 同进程 12 对 AB 的 B−A 中位 | 同进程 12 对 BA 的 B−A 中位 |
|---|---:|---:|---:|---:|
| [1,19] |−0.164|−0.107|−0.002|+0.039|
| [2,14] |−0.051|−0.021|+0.015|+0.036|
| 60 shape |−0.138|−0.085|−0.017|+0.083|

同进程 AB 的 B 胜出次数分别为 6、3、9 对，BA 为 5、3、1 对。72 对内两臂窗口首尾采得的 CPU 均相同；这项检查不覆盖窗口内从同一 CPU 短暂迁出后又返回的情况。独立进程较快的方向没有在同进程复现。安装版另一个进程的三类 Host 窗口中位为 3.368/3.415/3.430 µs。这个参考还包含整库构建和进程差异。[各窗口和长尾](analysis.json) · [原生 ELF 与构建环境](build-env.json)

私包反汇编显示 LL/SC 从共同的模式读取后顺序进入，LSE 跳转到独立代码块后返回。测得的是带真实 HWCAP 选择和该代码布局的提交路径。若要测处理器上两种原子指令的本体差，需另行控制分支布局和缓存线竞争；本试验没有将这些因素单独计时。[完整指令片段](instructions.txt)

增量补丁在指定公开提交的独立源文件副本上通过 `git apply --check`，保留源码 LF 结尾后，应用结果与实际编译单元的 SHA256 `9c223b09b4262782d2f834300a0678269750c65d4c2da5da4436c8598e364995` 完全一致。[逐文件复验](patch-verification.json)

## 训练语义

8 步原动态 eager 审计中，各腿从进程初始化至退出共 145,332 次目标递增，A/B 的另一条分支均为 0。每步梯度及 norm、最终 BF16/FP32 权重和 AdamW 状态通过原 runner 检查。实际训练输入的 8 步均完成有限值更新。另用真实 DeepSpeed BF16 ZeRO-2、原 01 overflow 和 AdamW 的单 rank 小模型，在梯度注入 +Inf、NaN、−Inf，再接有限步。两腿 7 步的跳步向量均为 [0,1,0,1,0,1,0]；逐步比较 loss、梯度、更新前后 BF16/FP32 权重与 AdamW 状态均一致。每个诊断 copy 触发一次本腿对应的原子递增。[原 HSTU 审计](training-audit.json) · [非有限值结果](overflow-summary.json) · [含哈希的七步原始记录及运行命令](overflow-evidence.tgz)

HSTU Level1 第 4、5 步按真实物理任务、HostToDevice flow、connection ID 和嵌套的 ACL API 复算。两腿分别关联到 17,214 和 17,215 个唯一 `aclrtLaunchKernel*` 或 `aclrtMemcpy*` 调用，每个 API 对应单笔设备工作。两步全量 P50 为 4.21→4.89、4.18→4.84 µs，P95 为 7.12→7.70、7.24→7.52 µs，P99 为 9.01→9.69、8.80→9.06 µs，最大值为 32.56→43.18、27.37→33.18 µs。按名义时钟，每笔 API 执行时间与其设备任务前全流空档的交集总和为 37.922→49.755、24.240→32.133 ms。跨域时钟会影响交集，单个 API 的 Host 持续时间取同一时钟。[逐物理任务复算](launch_tails.py) · [分算子、API 和物理流统计](launch-tails.json)

两腿的工作量按队列细分也相同。#4/#5 的物理流 47 各为 16,159/16,160 笔，流 44 均为 1,053 笔，流 45 均为 2 笔。其中 `aclrtLaunchKernelWithHostArgs` 各 15,093/15,094 次，Host P50 为 4.32→4.63、4.29→4.55 µs；`aclrtMemcpyAsync` 均为 2,121 次，P50 为 4.11→5.51、4.04→5.50 µs。这是一对正序 Level1 的观测。

## 尝试无优化路径

同一动态 eager runner、双向 embedding 和 `ACLNN_CACHE_LIMIT=65536` 下，两腿只切换 CANN 私有运行库中这一处 pending 原子。无 profiler 的第 10–59 步同步窗口按相同顺序处理 136 个样本、2,994 个 token。Level0 在相同配方下使用第 4–6 步的 [1,19]、[2,14]、[2,22]，Free 占比取 ΣFree/ΣStage。下表两行依次为正序和反序，各项时间单位 ms/步、提升为相对比例。

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
|361.994|373.592|−3.204%|425.727|88.666%|432.723|88.881%|−0.242%|
|364.558|373.233|−2.380%|470.367|89.748%|475.612|89.897%|−0.166%|

正序 A→B 的样本/s 为 7.514→7.281，token/s 为 165.417→160.282，Level0 Stage 为 480.145→486.857、Computing 为 54.306→54.026 ms/步。反序按同样的 A→B 口径，样本/s 为 7.461→7.288、token/s 为 164.254→160.436，Stage 为 524.099→529.063、Computing 为 53.651→53.337 ms/步。两次从 A 减去 B 的步时差为 −11.597 和 −8.675 ms，均值 −10.136 ms、样本标准差 2.066 ms。[窗口与八列明细](training-metrics.json)

第 4、5 步的 ATen 每线程顺序、各类队列项数量、ACL 调用次数、每条实际物理流上的设备工作及控制任务序列逐项相同。该实现未改善单次提交，两个训练方向均增加步时和 Free 占比。官方源码指令路径的结论覆盖本次 60 个合成形状；其余 CANN 版本、ISA 代码布局和业务形状分布未在此验证。[Level1 完整对照](training-level1.json) · [逐步统计与中间 CSV](training-level1-summary.tgz) · [无改动原训练 runner 的调用程序](training.py)

训练命令、worker 实际加载库、shape、原始 kernel_details/step_trace_time CSV、Level1 trace 和数值审计收据已归档。宿主归档目录为 `/mnt/paas/recsys-gr-ds019-host88-archive/dispatch-study-20261119-npu8/infra-only-followup70/analysis/cann-pending`，其中 `raw.tgz` SHA256 为 `6caf16b97431ee24dc040dbdd475e7a504153c645b3696a847ee9f18e807143c`，`summary.tgz` SHA256 为 `e493bef33970ac637e11774f003f89145e5556f0741731eba4b80fc757a6a1b8`。[本地训练原始包](training-raw.tgz)
