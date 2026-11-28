# PR 42634 计划下发对 score kernel 锁资源的支持

首版 causal score 前向的真实 ELF SHA256 为 `0e99dcac5419d11dc81a6d1f707eea70c6e7ed33263a5dfa4bc28cf48f99dfb2`，Triton 根据该编译结果生成并加载的 launcher 源码哈希为 `de772a68ff0e36d77446f13a1d8061e7f9b1d9e59166b2c26469ba364b5a171c`。它按顺序传入 FFTS、syncBlockLock、空 workspace、三个 Tensor 指针和 int32 的 X、Y、Z。物理 blockNum 与逻辑网格乘积相等，本 kernel 保留超过 40 的值。驱动为每次提交调用 workspace 分配接口取得 8 个 int64 的锁地址，用同步 `rtMemcpy` 把 64 字节零值写入。空锁槽的旧计划应拒绝这个 ABI。

本适配从已编译的 PR 绑定派生，保留原 `PackLaunch`、`SubmitLaunch`、同一个 `OpCommand` 及 `rtKernelLaunch`。给原布局的锁位置记录偏移，仅在经过 ABI 校验的计划内填入设备指针。consumer 在同一个回调内使用 Triton 原 `triton_allocate_sync_block_lock` 按真实 stream 分配、调用 `rtMemcpy`、下发 kernel、交还 `triton_release_retained_tensor`。每次提交取得新的 Tensor 持有句柄，队列中的回调保持动态库引用。句柄持有期和设备地址复用见下文。调用者在当前流上登记 Tensor 存储。每成功一次锁分配、H2D 初始化及 kernel 返回，分别累加一个 C++ 原子计数。每次锁初始化有一笔 64 字节 H2D API，原 kernel 提交数另计。[资源和回调](lock_resource.inc) · [H2D 与锁指针写入](lock_submit.inc) · [增量绑定构建](build.py)

构建使用训练镜像的独立 `--runtime=runc --network=none` 无卡容器，输出在 `/artifacts/dynamic-infra-20261127/pr42634/package-score-lock-v2`。实验保留三条路径 N 原生注意力、S 原 score Triton launcher、P 同一 ELF 由 PR 计划提交。先比较 60 种形状、含符号零和非有限输入的共 66 例，分别检查 N/S 和 S/P 的输出、dQ、dK、dV，并抽查 score、dScore。再对两条 stream、四个形状、四个同步窗口核对时间、C++ 成功计数和 Level1。[真实 ABI 核对及 dispatch](masked/locked.py) · [门禁与计时](masked/probe.py) · [构建回执](build-receipt.json)

目标后端把旧的 BF16 位型转无符号索引编译为负数。已终止旧二维调度器，其已启动的训练容器进程没有执行旧 PR 路径。首次尝试在资源核对时退出，下一次因设备外占在门前退出。当前隔离版本的前向与反向都用 int16 位型转 int32 后取低 16 位，限定查表下标为 0–65535。[当前 kernel](masked/kernels.py) · [当前 ABI 核对](masked/locked.py)

以下命令在宿主机 `/data/recsys-gr-ds019-host88/artifacts/dynamic-infra-20261127` 下运行。原 dispatcher 会检查设备映射、容器、共享锁和外部设备进程。

```bash
D=/artifacts/dynamic-infra-20261127
python3 -u dynamic-free70/wait_run.py score-lock-masked-v3 \
  python3 -B dynamic-free70/diagnose.py \
  --output 'dynamic-free70/score-lock-masked-v3-{attempt}' \
  --script dynamic-free70/score_lock_plan/masked/probe.py \
  --recipe dynamic-free70/score_lock_plan/masked/probe_recipe.json -- \
  --fold-dir "$D/dynamic-free70/pr42634_eager_fold" \
  --scores-dir "$D/dynamic-free70/score_lock_plan/masked" \
  --attention-dir "$D/dynamic-free70/recsdk-primitive-sources/causal_silu" \
  --schedule "$D/20261127T033119Z-baseline-l1-341x7gd3/workspace/configs/dynamic_shapes_npu8.json" \
  --output '{output}' --profile
```

## 2026-11-28 最后一轮与暂停位置

修复后实机重新编译的前向 ELF 为 `601733bd7f108ef4e92fd1a67466984607e5c67b55cba42c0998b69ddf04397f`，反向为 `185c1022ad2dfc3ff16db4de56cb21ae0f621e9dcc38d5c4c96231c2448608e5`。两者均报告 `lock_num=8`。共享锁门禁开跑前，逻辑设备 8 对应板 4 芯 0，健康且无设备进程。[ELF 哈希](receipts-masked-v3/masked-elf-sha256.txt) · [前向元数据](receipts-masked-v3/causal_score_forward.json) · [反向元数据](receipts-masked-v3/causal_score_backward.json) · [实机命令](receipts-masked-v3/command.json)

首个 finite [1,13] 中，N/S/P 完整 QKV 前后向都已返回到 Python，原生与 stock Triton 的整块输出比较显示零个不同 BF16 位型。随后程序提交三条路径的 raw score 和 dScore，还未写出本例的最终记录。两次只读进程采样均位于 `probe.py` 第 41 行、`bits` 第 51 行的首次 NS score 读回。AICore 持续 100%、AIVector 32%。读回会等待同流先前提交的多项工作，这段观测无法分辨具体哪个前向、反向或提交器造成停滞。[首次栈](receipts-masked-v3/pending-python.txt) · [复核栈](receipts-masked-v3/pending-python-2.txt)

实机 Triton 的 `npu_utils.cpp:358–372` 通过 `at_npu::native::allocate_workspace(size, stream)` 取设备地址，每次新建一个 Tensor 句柄，回调返回时删除句柄。对照本仓 PTA 的 `OpPreparation::unsafe_empty_workspace(size, stream)`、`DeviceWorkspaceAllocator::malloc`，后者以 stream 查找 workspace block，已有容量足够时可直接返回同一 `block->data_ptr`。本次同时使用 stock S 和计划 P；两条路径都使用同步 `rtMemcpy` 初始化 64 字节锁值，随后向请求的 stream 发起 kernel。此次没有直接记录各笔锁地址、设备端完成事件以及多次初始化的先后关系，尚无法确认同址提前初始化为原因。

10:54:41 对已核对身份且持续停滞的本实验进程发出 SIGTERM，退出码 143。停止时 66 例门禁、同步微窗、并行流压力与 Level1 均未完成；此项归入「尝试无优化路径／验证中止」。退出检查中该卡健康、设备进程为空、无孤立上下文。10:55:28 复核无设备进程，AICore/AIVector 均为 0%。双序训练未启动。[超时处置](receipts-masked-v3/own-timeout.txt) · [退出码](receipts-masked-v3/train.exit) · [结束后卡状态](receipts-masked-v3/after.card.json)
