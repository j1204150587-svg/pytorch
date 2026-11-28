# TorchNPU 只唤醒因队列已满而等待的 producer

目标为原始动态 eager HSTU，01 分块 overflow、Task Queue Mode 1、每设备共享队列。基线设备路径是 TorchNPU `38e483ee79ec8a3286517808740a811c0f0ae66c` 的 `Repository::Enqueue`、`ReadQueue` 和 `Dequeue`。原版成功入队后把 `write_idx.working` 置为 false。consumer 每消费一项，再检查 `!IsWriteWorking()`，成立就执行一次 `eventfd_write(efd_write,1)`。两次提交之间的正常空闲也满足条件。宿主往这个 fd 写入一个可读计数，不代表有线程因此醒来。

## 原训练的触发依据

原 01 在 #4 [1,19] 和 #5 [2,14] 各有 9,927 和 9,923 个可见 dequeue。其中 2,929、3,412 段覆盖前一服务结束到下一服务开始，全段没有任一 producer 的 enqueue 标记，后续请求已经入队。区间合计 7.222、11.519 ms，与官方 Free 相交 4.804、8.258 ms。同形再见的 #12/#13 分别有 1,661、2,259 段，与 Free 相交 5.343、10.793 ms。区间还包含回调释放、队列维护、调度及 profiler 负担，具体 syscall 次数由候选包统计。[首见原始统计](exposure.json) · [同形再见统计](exposure-repeat.json) · [复算](exposure.py)

原 !42634 的 `NPUPythonWrapperCodeGen` 在 Inductor 编译时生成 fast-launch callsite。原 01 动态 eager 正序的 B 腿有两个 C 扩展绑定，60 步均未导入目标 bind 模块、未创建 plan、未提交 plan。历史诊断在训练入口加了计数，这一收据仅作为历史记录。本轮候选通过现有 TorchNPU 入口自动执行。[既有原 PR 实机收据](../../dynamic-infra-20261127/pr42634/original-final/evidence/pr42634-original-no-profiler-ab/02-B/rank000/pr42634_receipt.json)

## 队列协议

新协议在 `WriteQueue` 报满后登记等待者，再使用原 fence 二次检查容量和退出状态。consumer 发布新的读指针并经过原 fence 后读取等待者计数。有等待者才向 `efd_write` 发放一枚唤醒额度。生产者的登记覆盖全部阻塞调用和重试出口，EINTR、状态变化均可注销。Mode 1 的 `efd_write` 使用 Linux `EFD_SEMAPHORE`，防止一名等待者一次耗尽多个唤醒额度。错误清理按当时登记人数唤醒，晚到的生产者复核退出状态。触发范围限制为每设备共享的 Mode 1。每流队列沿用原协议。同一设备使用独立的原子等待者计数，Repository 布局保持原 ABI。[候选补丁](producer-wake.patch) · [诊断和等待者计数](waiters.inc) · [安装器](install.py)

CPU-only runc 容器以真实 Linux eventfd 模拟满队列、EINTR、错误清理及多生产者队列。四轮测试分别使用 8 个 producer、12,000 个唯一提交，逐项验证无丢失与重复。每轮人工阻塞 8 名生产者并清理，均全部退出，另有 10,000 次低队深入队及消费，producer 通知数为 0。此项证据验证所改协议。实机训练语义和真实 Task Queue 的全部执行仍需独立验证。[测试源](wake_stress.cpp) · [四轮日志](cpu-stress.jsonl)

## 构建、开关与复算

独立构建仅编译 `NPUQueue.cpp` 并重链 `libtorch_npu.so`。v2 实验库 SHA-256 为 `2c3f257319d88b14bd628de41d74dec511ce0e00ea4de14baa64b2d5cd9e2690`。宿主原始构建在 `/data/recsys-gr-ds019-host88/artifacts/infra-only-20261128/queue/build-v2`。正式卡上由宿主配方复制该 so 到原训练 wheel 的 Python-identical overlay。两腿共同核验业务代码、形状与模型文件哈希、加载路径及库哈希。[构建命令与对象哈希](build-receipt.json) · [构建器](build.py)

- A/B 唯一机制开关是 `TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY=0/1`。进程初始化在 stderr 写一次 `PTA_QUEUE_PRODUCER_WAKE_CONFIG`，记录设备、机制与诊断开关。
- 激活诊断两腿共用 `TORCH_NPU_QUEUE_WAKE_AUDIT=1`。进程退出时输出 dequeue 数、满队列登记与读、成功的正常通知、通知前未观察到等待者的次数、清理及在册等待者数。判断通知时的计数是并发快照，有并发满队列时应结合重试次数理解。
- 正式同步窗与 Level0/Level1 两腿共用 `TORCH_NPU_QUEUE_WAKE_AUDIT=0`。这一路径无统计原子更新，A 沿用原判断，不访问等待者计数，B 只执行协议实际需要的等待者存取。
- 若开启诊断时没有减少通知，或仅因附加打点得到改善，则候选机制未验证。要归因单次提交，需同形、ATen、流内 kernel/copy 次数及顺序相同，并以裸计时和两序官方 Free 各自给结论。

原始训练入口的八步实机审计已经完成。A/B 从进程开始至结束均处理了 87,845 项，两腿的队列满登记、等待、清理及最终等待者数均为 0。A 对没有登记等待者的 `efd_write` 成功写入 86,874 次，B 正常通知写入为 0。进程总数含初始化与收尾。两腿记录的梯度及范数、BF16 权重、FP32 主权重、AdamW 一阶和二阶矩、更新计数完全一致，AdamW 均执行了 8 次更新。此轮没有触发 overflow 跳步。[实机队列、语义与库的回执](gate-and-counters.json)

下面的参数是一次完整 Level1 配对的原始目录、独立分析输出目录，以及此前保存的 HostToDevice 和 correlation 联结脚本。分析通道只读原 trace、metrics 和 CSV，在输出目录中保存两腿的逐笔事件和对照。

```sh
python3 -B analyze_pair.py "$RUN" "$OUTPUT" --analyzer "$STUDY/analysis/analyze_dynamic_trace.py"
```

脚本保存每个 shape 的参数准备、入队、队列等待、consumer 服务、单笔 ACL launch 分位数，单列长尾及与已知后继设备工作的交集。每个物理流比较实际工作序列和完整控制事件序列。[配对复算](analyze_pair.py)

## 逻辑 NPU 8 的动态训练结果

上行为 A/B，下行为 B/A，每行均按 A 为优化前、B 为优化后排列。端到端提升比例取 (A−B)/A，Free 占比取三步 ΣFree/ΣStage，最后一列取 (A 占比−B 占比)/A 占比。

| 优化前端到端 ms/步 | 优化后端到端 ms/步 | 端到端提升比例 | 优化前 Free ms/步 | 优化前 Free 占比 | 优化后 Free ms/步 | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 578.309 | 569.639 | 1.499% | 615.817 | 92.303% | 561.133 | 91.539% | 0.827% |
| 604.269 | 576.804 | 4.545% | 574.567 | 91.621% | 589.345 | 92.028% | −0.445% |

相同 60 个合成实际形状覆盖 batch 1–4、序列长 12–32。首 10 步预热，首尾同步的 50 步含 136 样本、2,994 token。A/B 的样本/s 为 4.7034→4.7750，token/s 为 103.543→105.119。B/A 的样本/s 为 4.5013→4.7156，token/s 为 99.095→103.813。两对各节省 8.670 和 27.465 ms/步，样本标准差 13.290 ms/步，仅有两对。Level0 的 #4–6 为 [1,19]、[2,14]、[2,22]。A/B 的 Stage 667.167→612.996、Computing 51.214→51.741 ms/步，B/A 的 Stage 627.115→640.394、Computing 52.435→50.942 ms/步。两对端到端同向，Free 占比变化异向。[计时与原始运行路径](paired-endpoints.json)

Level1 正序逐步联结的 44、45、47 三个物理流，其设备工作与控制事件的完整有序签名完全相同。#4、#5 的 ACL launch 各为 15,093、15,094，每步同步 memcpy 6 次、异步 memcpy 2,121 次。物理工作数分流各为 1,053/8/16,159 和 1,053/8/16,160。两腿也有相同的 ATen、enqueue 次数及各生产线程的 ATen 顺序。[逐步与逐算子结果](level1-pair.json) · [逐笔 CSV 和配对 JSON](queue-l1-linked.tar.gz)

| Level1 指标 | #4 [1,19]，A→B | #5 [2,14]，A→B |
|---|---:|---:|
| ATen 至 enqueue、enqueue 本身 p95，µs | 69.60→70.16，8.20→7.54 | 69.55→71.42，8.35→7.85 |
| 可见等待 p95 / p99 / 最大，µs | 35.93→36.46 / 48.28→47.77 / 1344.96→1180.87 | 37.07→37.56 / 50.26→51.89 / 1329.04→1136.44 |
| consumer 服务 p95，µs | 121.73→127.27 | 122.45→127.99 |
| 单次 ACL kernel launch p95，µs | 7.35→7.85 | 7.39→7.82 |
| 后继设备空档与可见队列等待的重合，ms | 12.324→11.617 | 10.909→10.407 |
| 后继已入队、无 producer 入队的相邻 dequeue 间隙数 | 3,149→3,180 | 3,195→3,198 |
| 上述间隙时长及其中与官方 Free 的交集，ms | 8.227→3.544，5.302→2.271 | 8.648→3.860，6.529→2.880 |

算子层面，每步 1,027 个反向 embedding 的 consumer p95 分别从 141.88→144.99、143.07→147.99 µs，每步 3,218 个 InplaceCopy 的 consumer p95 从 11.29→11.48、12.01→13.00 µs。两步实际单笔 ACL launch p95 均上升。能直接验证的是去除了无等待者的系统调用，并缩短了下一项已就绪时的可见出队间隔。Level1 未标出每次 eventfd 的进入/退出，间隔中也包含 release 和其余队列维护。局部 Free 交集分别减少 3.031 和 3.649 ms，不能据此解释整步节省的全部 8.670–27.465 ms，亦未建立全程单笔 ACL 提速或可重复的三步 Free 改善。
