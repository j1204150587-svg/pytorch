# 侧表 BF16 Add 的 CANN executor 重用

## 准入证据

两腿共同使用侧表前向和反向 embedding 描述符规范化、CANN cache 65536、Task Queue Mode 1 和动态 eager。以已有双向 embedding 的真机 Level1 为参照，在主线程按连续的侧表 embedding、mean、add 队列项定位功能调用。第 4、5 步实际 shape 为 [1,19] 和 [2,14]。每步分别有 1,023 个侧表累加，均为物理流 47 上单次 `aclnnAdd_AddAiCore_Add`。两步总共只观测到一次 `aclnnAddGetWorkspaceSize`。[原始定位及复算](admission.py) · [聚合数据](admission.json)

下表为同一 CPU 阶段与全流空档求交后的时长，每步单位 ms。表内 producer 与 consumer 可以并行，不能将各行相加作为可回收时间。括号为后继任务恰为任一 Add 时的空档交集。CPU producer 的前半段与其他算子的设备工作交错，单看紧邻 Add 的空档会遗漏这些区间。

| 区间 | [1,19] | [2,14] |
|---|---:|---:|
| aten::add 入口至 enqueue，含原生输出分配 |4.809（0.119）|8.728（0.023）|
| 其中输出分配 empty_tensor |1.209（0.016）|2.280（0.005）|
| 分配以外的 producer 可见上界 |3.600|6.447|
| enqueue |1.703（0.335）|2.434（0.014）|
| dequeue 至 aclrtLaunchKernel |1.787（0.922）|2.742（0.078）|
| aclrtLaunchKernel |3.580（2.672）|4.948（0.292）|
| launch 返回至 dequeue 结束 |1.365（0.991）|2.635（0.511）|

[1,19] 的首个侧表 Add 已无 workspace 查询。其余 1,022 次中，producer、队列等待、consumer 和 ACL launch 的 P50/P99 分别为 8.38/16.73、2.45/28.60、8.97/10.64、4.82/5.86 µs。[2,14] 首个侧表 Add 有一次 workspace 查询，producer 36.64 µs、consumer 57.30 µs、ACL launch 7.23 µs，对应本笔设备任务前的全流空档为 42.80 µs。后续 1,022 次均无 workspace 查询，四项 P50/P99 分别为 8.69/16.99、13.835/58.11、11.75/18.63、5.48/11.30 µs。此步最长的 103.96 µs 队列等待只对应 11.42 µs 的直接设备空档。[逐次原始数据](side-add-requests.csv.gz)

## 机制和证伪条件

仅修改 functional `op_api::add(const Tensor&, const Tensor&, const Scalar&)` 的提交段。A 完整调用 stock `EXEC_NPU_CMD(aclnnAdd)`，B 在匹配的连续 [B,L,64] BF16、alpha=1、单设备/流和受限加载条件下，从 consumer 初始化该 shape 的可重复 CANN executor，每次将当前两个输入和独立输出的真实地址绑定到原描述符，执行原 `aclnnAdd`。保留 ATen 输出申请、原 stream、队列、单次 launch 和 named tensor 传播。对于两个流或生命周期不同的张量，producer 记录真实 stream 并持有三个 Tensor 至提交完成。不同 owner stream 或不支持布局走 stock。绑定失败时本次在 stage 2 前转 single-use，后续调用回 stock。正式计时关闭诊断开关。[原生代码](repeat_add.inc) · [同一 ELF 的安装方式](build.py)

同库叠加在已校验的双向 embedding 对象之上，ELF SHA256 为 `9a99b4a695e2f93b0dafdbf03263d7f145c211927eea35aa851bd42e2180f4d8`。增量补丁在 pristine op-plugin commit `eaf6b55812a1998a4a40e485e3b2861a40e9c2f6` 的 Add 文件上验证可应用，实际重链时包含双向 embedding 的 3 个共用对象。[Add 增量补丁](from-stock.patch) · [逐文件 SHA](patch-verification.json) · [重链命令和最终收据](build-receipt.json) · [共同前置补丁](../embedding/embedding.patch)

先测 60 个 shape、同时存活的不同 operand 和 output 地址、BF16 特殊值、并发双 stream、提前释放输入和 stage 2 前的模拟绑定失败。若 CANN 拒绝 executor 重用或逐位结果不符，停止性能试验。两腿使用原 eager functional API，在新进程先按相同 shape 和次序暖缓存。每窗首尾同步，对 [1,19]、[2,14] 各测 512 次，再对 60 个 shape 按原顺序反复 16 轮；各取 9 个窗口中位数，各测一次正、反序。若三个 workload 在两个顺序全部回退，则从训练候选中撤下。另用同进程和同一 stream 的 Level1 标记核验两个 shape 的两臂各 8 次提交、实际工作名称、延迟和缓存热度。[运行脚本](micro.py) · [同流复算](analyze_level1.py)

## 尝试无优化路径

同一逻辑 NPU 8，两个方向的所有窗口均回退。下表是单次 Add 同步窗口的均摊时间，单位 µs。

| 实际形状 | 正序 A | 正序 B | B−A | 反序 A | 反序 B | B−A |
|---|---:|---:|---:|---:|---:|---:|
| [1,19] |12.034|20.177|+8.142|12.728|22.768|+10.040|
| [2,14] |11.407|20.287|+8.880|13.552|22.458|+8.906|
| 按序轮换 60 个 shape |11.417|20.257|+8.840|12.790|22.139|+9.348|

60 个实际 shape 中，有 60 个新 plan、186 次真实地址重绑定和 246 次 stage2 launch。非 owner 的 24 笔并发请求及 3 笔布局/alpha 回退均逐位通过。模拟首次重绑定失败时，计数为一次 single-use，4 笔后续调用回 stock；该进程共 6 次输出逐位一致。整个微门结束后卡上无任务进程。[运行汇总](micro.json) · [错误回退](part-1-fault.json) · [挂载、进程、命令、原始窗口、CANN CSV 和 trace](micro-evidence.tgz)

单进程 Level1 的四个标记每个均为 8 个 ATen、原物理流 47 的 8 个 Add 和 0 次 workspace 查询。[1,19] producer 到 enqueue 中位数由 8.995 升至 21.015 µs，consumer 由 9.68 升至 10.67 µs，ACL launch 由 5.275 降至 5.120 µs。[2,14] 分别为 9.61→15.58、8.51→10.54、4.455→5.170 µs。新增 guard、allocator stream 登记、Tensor 持有及队列回调属于整体替换，未逐项独立定时。已热缓存上，候选未缩短单次下发，按预先写明的微门停止，未占用整步 E2E 和 Free 配对。[逐笔、P50、P95 与最大值](profile-analysis.json)
