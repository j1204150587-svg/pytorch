# isDeviceCtxActive 非兼容路径的已激活设备位图

基线是原动态 eager HSTU、01 overflow、双向 embedding 和 65,536 项 executor 缓存。按进程的 ELF 映射独立复算两份相同配置的 8 秒 CPU 采样。两个运行分别有 2,156、2,103 个样本，其中 61、62 个栈经过 isDeviceCtxActive，60、62 个经过 MaybeSetDevice。37、42 个命中栈位于保留的 CheckCompatibleImpl 检查里面，只有 4、5 个命中栈以 pthread_mutex_lock 为叶函数。这些采样记录在 CPU 上执行的时刻，不能测出睡眠等待互斥锁的时长。[原始 perf](../followup70/host-samples) · [逐栈复算](perf_context.py) · [详细结果](perf-context.json)

仅修改 pinned PTA 的 `torch_npu/csrc/core/npu/NPUFunctions.cpp`。查询先执行原 CheckCompatibleImpl，兼容模式继续调用 hasPrimaryContext。在非兼容模式，设备号 0–63 可以查询一个无锁正命中位；未命中、负数、越界仍取得原递归锁，按原 int8 表键语义查询。只在原锁内读到非空上下文才发布一个位。ResetUsedDevices 每次取得原锁立即交换清零；重置期间的同线程递归读取仍按原表返回，但不能重新发布。异常或部分重置失败后，原有表仍是后续查询的依据。证伪条件为训练未走该分支、原生锁查找没有减少，或两序训练窗口及 Level0 没有可重复收益。候选 SHA-256 为 `43ee422ed89a21055bd724dd8562d89513a9be99a4dfff0b939af0388d3ec8b4`。[完整源码 diff](context-state.patch) · [构建收据](build-receipt.json)

### 正确性与函数计时

CPU-only 门测由原始版本的 SetDevice、LazySetDevice、MaybeSetDevice、ResetUsedDevices、hasPrimaryContext、isDeviceCtxActive 完整函数体编译，替换外部 ACL ABI。A/B 均覆盖空表、空指针/错误、设备 0/63/64、负数和越界键、兼容查询和非法设备异常、重置中同线程递归读、其他线程读被阻塞、部分失败及异常清理。没有在真实设备执行 reset。[门测程序](mock_gate.py) · [门测结果](mock.json)

实卡单设备上另由 4 个线程各执行 512 次原库内 `isDeviceCtxActive(0)`。非兼容 A 的 2,048 次均查原锁表，B 的 2,048 次均命中正位；在兼容模式两个版本均调用 hasPrimaryContext 2,048 次。独立微测同时确认未初始化设备不返回真。[设备微门](device_micro.py)

关闭内部计数与 profiler 后，每次用本机 C 函数调用原生入口 2,048 次，再取各批平均 ns/次的中位数。单线程、2 线程、4 线程的 A/B/B/A 读数如下。该计时包括保留的兼容模式检查，不包含 Python 逐调用开销。[计时器](time_calls.c) · [并发与采样配置](time_calls.py)

| 并发线程 | A 第一次 | B 第一次 | B 第二次 | A 第二次 |
|---:|---:|---:|---:|---:|
| 1 | 92.52 | 66.49 | 67.62 | 90.65 |
| 2 | 218.87 | 66.72 | 68.08 | 162.02 |
| 4 | 667.40 | 66.11 | 68.04 | 562.84 |

独立小模型的 BF16 ZeRO-2 真实非有限状态验证通过。七个步骤依次 finite、+Inf、finite、NaN、finite、−Inf、finite，四个有限步更新、三个异常步保持 BF16 权重、FP32 master 和 AdamW 状态，后续有限步恢复更新。A/B 逐步 loss、梯度和状态签名完全一致；B 的 1,935 个上下文检查中，1,934 次正位命中、1 次查表并发布。[独立验证代码](overflow_probe.py)

原动态 HSTU 8 步审计的训练进程，两腿都调用了 isDeviceCtxActive 446,559 次。A 全部运行原锁表分支，B 446,558 次位图命中、1 次查表并发布正位，兼容分支为零。该计数覆盖进程初始化及审计辅助调用。两腿梯度、norm、BF16 权重、FP32 master 及 AdamW 状态一致，loss 最大绝对差 1.907×10⁻⁶；8 个有限步骤正常更新。[原始训练审计](evidence/pairs/context-state-audit-ab/result.json) · [各设备微门与异常步骤记录](evidence/probes)

### Level1 设备提交与主机阶段

第 4 步 [1,19] 和第 5 步 [2,14] 的物理设备提交数、设备流、槽位及任务全序列相同。B 的 host destroy_event 标记分别少 6、1 个。每步 3,217 个带设备工作的 InplaceCopy 可按 ATen 序号和流槽匹配，两步 producer 到 enqueue 的 p95 为 31.59→32.78、32.00→33.44 µs，对应后继空档之和 162.17→162.17、157.85→163.60 ms。每步 1,027 个 EmbeddingDenseBackward 匹配调用的 producer p95 为 25.74→22.09、26.87→21.07 µs，但其 consumer p95 同时从 51.89→77.23、54.23→76.80 µs。实际设备序列一致，目标函数的独立成本下降；两步时间线没有显示各操作普遍缩短单次提交。单次 isDeviceCtxActive 在 Level1 没有独立 marker，无法将每次位图命中与某一个设备空档直接相连。[物理提交核对](level1/submissions_compare.json) · [七种常见 ACLNN 操作的分位数、长尾和空档](level1/paired_phases.json)

### 原动态 eager HSTU 的正反序窗口

原 runner 的相同 60 个合成形状、相同输入顺序。所有无 profiler 腿均预热 10 步，在同步窗口测量后续 50 步，共 136 个样本、2,994 个 token。Level0 统计第 4–6 步 [1,19]、[2,14]、[2,22] 的 ΣFree/ΣStage。执行顺序分列如下，B/A 表中各列仍依次报告 A、B。反序 Level0 在等候共享锁时已撤销，两腿均未开始。[数值与各腿原始目录](analysis/paired.json) · [复算程序](partial_metrics.py)

| 优化前端到端 ms/步 | 优化后端到端 ms/步 | 端到端提升 | 优化前 Free ms/步 | 优化前 Free 占比 | 优化后 Free ms/步 | 优化后 Free 占比 | Free 占比提升 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 414.631 | 368.694 | +11.079% | 601.224 | 91.722% | 684.107 | 92.689% | −1.054% |
| 401.699 | 414.642 | −3.222% | 未测 | 未测 | 未测 | 未测 | 未测 |

| 顺序 | 腿 | 样本/s | token/s | Stage ms/步 | Computing ms/步 |
|---|---|---:|---:|---:|---:|
| A/B | A | 6.560 | 144.417 | 655.482 | 54.139 |
| A/B | B | 7.377 | 162.411 | 738.069 | 53.758 |
| B/A | A | 6.771 | 149.067 | 未测 | 未测 |
| B/A | B | 6.560 | 144.414 | 未测 | 未测 |

A−B 的两组差分别为 +45.937、−12.943 ms/步，均值 +16.497，样本标准差 41.635 ms/步。单函数在无 profiler 的实卡微测中确实减少服务耗时；原训练的正反序端到端变化方向相反，且正序空档占比上升。归「尝试无优化路径」。统计覆盖合成 batch 1–4、序列长 12–32；实际设备时间线只覆盖已列的三个 Level0 步、两个 Level1 步。

已核对审计 A/B、无 profiler A/B 与 B/A、Level0 A/B、Level1 A/B 五组的十条训练腿和二十个腿前、腿后卡况快照。二十次均为逻辑 8、板 4/chip 0、物理 8、健康 OK，目标 chip 的 PID 列为空且无孤立上下文。同板其他 chip 的活动保存在快照中。三份原始包分别含原 runner 命令、worker PID 和载入库、卡况、指标或原始 trace。对应哈希见清单。[卡况索引](analysis/card_receipts.json) · [八步训练与门测原包](context-audit.tgz) · [无 profiler 与正序 Level0 原包](context-performance-partial.tgz) · [Level1 两腿原包](context-l1.tgz)
