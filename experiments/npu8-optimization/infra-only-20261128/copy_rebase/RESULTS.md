# 动态 eager HSTU 的 InplaceCopy 实机复测

所有训练腿使用原始 HSTU、原始 ZeRO-2、动态 60-shape schedule、1,024 张侧表、01 分块 overflow、Task Queue Mode 1。两腿共同启用 `ACLNN_CACHE_LIMIT=65536` 和已验证的双向 embedding index 规范化。本目录的候选只切换 PTA 原生代码。地址记录、逐类计数和带非有限值的小模型仅用于独立审计。

## 512B 整对齐描述符

`copy_d2d_baseformat_opapi` 的原 overlap 检查之后，对满足设备、dtype、别名、步长、基格式、存储范围与 512B 对齐条件的地址，在单次 ACLNN 参数序列化与转换时缩小 descriptor 的形式存储区间。复用原 ACLNN、任务流及 allocator 生命周期。启动依据是基准第 4、5 步流 44 各有 1,052 笔单 SDMA 且逐笔调用 workspace。若典型实际操作不通过 guard，或同 key 无法命中 executor，则这一缓存机制不成立。

25 项算子微测通过，8 步原运行语义一致。实际 25,746 个 copy 只有 48 次满足 guard，25,698 次回退。96 个参数 key 全部重建 descriptor。仅已开始的正序同步窗口为 396.942→388.830 ms/步、6.852→6.995 样本/s、150.853→154.001 token/s。因真正目标形态未命中，列入「尝试无优化路径」。单次正序数字未作为收益。库 SHA-256 为 `19122bc7ee92346fe6f59528b4301fe15e1146d92c1257d872431021f9d5367f`。[构建与代码](build-receipt.json) · [原始两腿](exact-evidence/pairs) · [原生 guard 分布](guard-evidence/summary.json)

## 128B 残差描述符

guard 诊断表明 BF16 `[4096]` 的实际目标地址总在 512B 基址后的第 128 字节，原始训练每步有 2×1,024 笔此类来源。修订版将 base 按 512B 向下对齐，保留 0 或 64 个 BF16 元素的逻辑偏移，storage_dims 止于有效数据末端。禁止 alias、非连续、跨设备、异 dtype、非基格式；原 `CheckMemoryOverLaps` 和 `aclnnInplaceCopy` 继续执行。37 项原始 BF16 bit、保护区、不同存储 owner 和跨流微测在两腿通过。[实现](copy_scope_128.inc) · [构建](build-128-receipt.json) · [8 步原 runner 审计](residual128-evidence/pairs/copy-rebase-128-audit-ab/result.json)

8 步 B 的 16,384 次目标调用全部激活，9,362 次其他调用回退，32,768 次参数转换全部新建描述符。单 SDMA 的 CANN 内核注册会显式将当前普通 executor 缓存标为无效。[运行库路径与反汇编原件](CANN.md)

正序 Level1 的第 4 步 [1,19] 和第 5 步 [2,14]，两腿分别有 1,052 笔流 44 和 1,060 笔流 47 的单 SDMA，ATen 顺序、物理流、提交槽、CANN launch API 和设备任务类别均可逐笔匹配。流 44 两步 GetWorkspaceSize 共耗时 6.93→5.94、6.92→5.93 ms，流 47 则为 12.25→12.80、12.29→12.80 ms，所有匹配请求仍准备 workspace。流 44 producer 到 enqueue 的 p95 分别为 20.93→16.70 和 21.20→16.69 µs，实际 ACL memcpy p95 为 5.21→5.12 和 5.28→5.17 µs。候选没有减少提交或绕过 executor 准备。第 4 步另有一个 host destroy_event 标记差异，该步物理提交全序列相同。[提交对照](level1-128/submissions_compare.json) · [逐物理槽延迟、长尾及空档对应](level1-128/single_sdma.json)

按 8 步独立地址审计重复出现的流 44 长度序列，映射每步 1,052 个物理提交中第 26–1,049 项为 1,024 笔 BF16 [4096]。第 4/5 步这些目标调用的 producer p95 为 20.86→16.64、21.00→16.62 µs。相同流上 28 笔未命中候选的形态也从 20.64→18.42、25.77→17.73 µs，workspace p50 同时下降。正序 profile 的公共波动限制了逐笔改进的归因。每步首次 [4096] 与其余 1,023 笔均进入 workspace；此前 HSTU 步已使用 [4096]。60 个互不相同的整步 shape 无法从这个窗口估计同一整步 shape 的首次与复用差异。[实际子形态与逐步分位数](level1-128/stream44_shapes.json)

128B 残差版本列入「尝试无优化路径」。两序端到端 −4.592%、+1.812%，A−B 均值 −4.770 ms/步，样本标准差 16.837 ms/步。Free 占比相对变化 −0.079%、+0.472%。两个指标均未显示可重复改善。[两序八列、吞吐与 Stage/Computing](residual128-analysis/PAIRS.md) · [精确数值](residual128-analysis/paired.json) · [native 修改](residual128.patch)

## 在 TQ Mode 1 中局部移动单次 copy 的准备

另一项实现保留原始 `aclnnInplaceCopy`、ACL runtime 提交与单 SDMA，只将 BF16 `[4096]` 目标的 `CopyTypesV2` 快照交给 `OpCommand::RunOpApiV2` 的 consumer，在 consumer 内执行 descriptor 和 workspace 准备。目标 wheel 的 `TensorStruct` 仅拷贝尺寸、步长与裸地址，闭包额外持有两个 Storage。原生缓存 allocator 仅在两端所有者的 allocation stream 均与当前提交 stream 一致时放行，跨流、外部 allocator、图 capture、非目标元数据和 alias 都回原路径。这样可用同流 FIFO 约束内存重用，不需要插入设备事件。若源宿 guard 使训练主体不激活、consumer 服务导致总延迟转移、或无 profiler 双序与 Level0 没有稳定改善，则该方案不能列为有效收益。[构建与来源](build_consumer.py) · [guard](copy_consumer.inc) · [构建收据](build-consumer-receipt.json)

独立算子微测两腿各 14 项，B 8 次命中和 6 次回退，8 次参数快照都在 consumer 执行。覆盖两种分配流、4 种偏移、producer 改写 view 元数据/切换存储、释放源后复用 allocator、跨流回退、部分重叠错误。

原 HSTU 8 步共有 25,746 笔 copy，B 8,192 次满足条件、17,554 次回退，8,192 次快照全部由 consumer 执行。两腿实际梯度、norm、BF16 权重、FP32 master 和 AdamW 状态一致，loss 最大绝对差 3.815×10⁻⁶。每步 2,048 个 [4096] BF16 单 SDMA 中，该 guard 允许 1,024 个走新路径。[原始审计](consumer-audit.json)

Level1 第 4、5 步每步都将流 47 的 1,024 个 workspace 从 producer 移到 consumer。两个 stream 共 2,112 笔单 SDMA/步全部匹配 ATen 次序、流、设备槽和 ACL API，物理工作和提交数量一致。目标 producer→enqueue p95 为 36.33→10.92、35.50→11.38 µs。原 producer workspace 总耗时 12.274、11.519 ms，对应新 consumer workspace 时长 4.872、4.754 ms。workspace 在后继设备空档内的实际交集从 8.405→4.767、5.090→2.565 ms。目标后继已关联空档从 62.07→34.68、59.20→34.65 ms。相同操作的 dequeue 总服务分别 23.06→19.35、14.95→19.12 ms；第二步明确观察到一部分主机服务移到了 consumer。两步 ACL memcpy p95 从 8.53→5.64、7.46→5.63 µs，跨腿 host 公共波动使此项不能单独归给该改动。[移动请求的逐步分位数及空档](level1-consumer/moved_workspace.json) · [完整逐槽提交核对](level1-consumer/single_sdma.json) · [全步序列](level1-consumer/submissions_compare.json)

标准两序 A/B、B/A 的端到端降幅为 11.469%、0.489%，A−B 差分别为 46.028、1.852 ms/步，均值 23.940，样本标准差 31.237 ms/步。Free 占比相对改善 0.064%、0.833%，绝对 Free 分别从 438.366→446.199、486.803→443.453 ms/步。单次 producer 准备确实变短；全步 Free 的绝对时间存在顺序波动。[两序标准八列、吞吐、Stage、Computing](consumer-analysis/PAIRS.md) · [实际构建差异](consumer.patch)

鉴于两序耗时差异较大，再用同库、相同 recipe 做一个反序和一个正序确认。额外 B/A 按 A→B 计为 390.863→405.967 ms/步（−3.864%），额外 A/B 为 418.135→378.657 ms/步（+9.442%）。四对 A−B 平均 +18.063 ms/步、中位数 +20.665、样本标准差 29.459，范围 −15.105 至 +46.028 ms。两次 A/B 均明显较快，B/A 一次接近持平、一次回退，显示顺序相关波动。该路径归入「尝试无优化路径」的端到端未稳定复现项；保留已证实的局部参数准备移位及相同物理提交的耗时。[四对端到端及每腿样本/s、token/s](consumer-analysis/confirmation.json) · [确认窗口原始回执](consumer-confirm.tgz)

独立的小模型使用真实 DeepSpeed 0.19.0 BF16 ZeRO-2、原 01 overflow 和 AdamW。两个候选在各自的 A/B 使用同一个单 rank 脚本，将经过目标 BF16 copy 的梯度交给真实 optimizer。按 finite、+Inf、finite、NaN、finite、−Inf、finite 的顺序，跳步向量四腿均为 0、1、0、1、0、1、0。两个 B 的目标操作各命中 7/7，每腿 7 次原 01 检测均走 fast path。逐步 loss、梯度 SHA、BF16 权重、FP32 master、AdamW 状态在同库 A/B 完全一致，三个跳步均保持状态，随后 finite step 继续更新。原 HSTU 8 步均为 finite。这项强制异常验证与正式训练、计时独立。[独立脚本](overflow_probe.py) · [四腿原始命令、卡检查和逐步状态](overflow-evidence/overflow)

## 已核查的其他路径

同址长区间已经不产生设备提交。流 44 后续 1,052 笔的地址两端在每步全部同向相邻，但单次 dequeue 开始时已发布的连续请求，第 4 步理论上限仅省 6 次，第 5 步为 0；流 47 同类段源端相邻率为零。[地址与发布顺序证据](ADDRESS.md)

时序模型在首笔发布后等待 1、2、5、10 或 20 µs 均未新增可合并的提交。50 µs 才可能合并 175/190 笔，原可省 ACLNN 服务上限为 1.290/1.335 ms，组首累计新增等待约 40.71/41.09 ms。当前两步的证据不支持为此实现延迟刷新的新设备队列。[有界等待、屏障和成本拆分](LOOKAHEAD.md)
