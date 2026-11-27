# PR 42634 与动态 eager ordered_64

PR 的实现提交为 `b1c30ebd929bd381fdb14278657bb7b3da692701`，head `2701f7b3e8d89625752edbde10e42c2b2172272c` 补充使用说明。官方 `TORCHINDUCTOR_NPU_FAST_LAUNCH=1` 在导入前生效，入口是 NPU Inductor Python Wrapper。普通 eager 的 ACLNN 操作和直接的 `ordered_64[grid](...)` 都没有 Wrapper 绑定；直接套用环境变量时应分别汇报零个计划和零次命中。[PR](https://gitcode.com/Ascend/pytorch/pull/42634/commit)

## 唯一已定位的 eager Triton 调用

Direct B 的 #4–6 每步有一次原始 `ordered_64`。同一步共用单个物理 stream 47；本核前最近的硬件任务都是 `aclnnMean_ReduceMeanAiCore_ReduceMean`。三步的 OrderedTableSum 入口至入队分别为 224.09 / 204.53 / 236.71 µs，其中和前序 ReduceMean 结束至本核开始之间真实空档相交 200.065 / 159.495 / 167.965 µs。范围内还包括输出分配等工作。原入队 9.52–10.99 µs、可见排队等待 2.40–3.38 µs、consumer 服务 15.13–17.05 µs；核时为 71.38 / 71.40 / 143.20 µs。[逐步预算](fold-budget.json) · [复算](fold_budget.py)

## 首轮实机门

1. 在逻辑 8 的专有映射和共享锁内运行下面的命令。复用原来的 `npu_ordered_fold.py`。A 和 B 使用同一份编译二进制的两指针 ABI，T=1024 为 constexpr，真实动态 grid 为 (B×L,1,1)。普通 eager 先成功运行，拿回 CompiledKernel 验资源、签名、stub、FFTS 和设备。保留真实 owner，检查同形状换地址和跨 shape 的 launcher 身份。原生符号存在才能使用现有 wheel；如使用按 PR 构建的独立扩展，用 `--bindings-module` 明示模块并保留构建记录。
2. 覆盖 [1,19]、[2,14]、[2,22]、[4,31] 和重复 [2,14]，每次实际重分配输入/输出，交替两条真实 stream。对照同输入 stock eager 的全部 BF16 位，再用 PyTorch CPU autograd profiler 验证一次动态回退。记录 C++ 入口和成功返回的不同计数；进入 C++ 后抛错应保留异常并停止。此后四轮 AB/BA 的每轮 32 次，留逐笔 Host 延迟及窗口首尾同步的完成时间。若环境变量、plan 和 eager 适配耗时需要分开归因，应额外比对同一二进制的普通 launcher，保持两腿分配和 stream 获取一致。
3. 完整训练的 A/B 都保留已验证的有序侧表 fold 和同层所有前置项，仅切换 eager 适配器。原始全 ATen 基线没有该 fold。按步记录计划建立、planned、按原因回退、PyTorch profiler bool、Triton hooks 数；固定真实 60 shape 序列、种子、1024 张逻辑表，核对 loss、梯度、更新、overflow 注入和跳步。NPU Level0 使用 torch_npu.profiler 时逐步实测实际选择的入口；若 #4–6 均回退，记录回退腿的 Free，不能据此声称快路径改变 Free。提交减少与每次加快应分别依据相同算子、物理 stream 和真实提交数判断。

```bash
TASK_QUEUE_ENABLE=1 TORCHINDUCTOR_NPU_FAST_LAUNCH=1 \
  python fold_probe.py --fold-dir /path/to/side_tables --out fold-pr42634.json
```

## 目标 wheel 的原始 ABI 和训练接入

目标为 torch_npu 2.10.0.post6、Triton 3.2.0。原编译器只在 callback 存在时将 workspace 和 lock 数写入 metadata。探针从实际 NPULauncher 重生 C++ 源，利用与 wheel 相同的 SHA256(header + NUL + wrapper) 查原始加载的 .so。原 launcher 的 .so 摘要为 `bc75aa433ade752004148ba2c339728781ac45386581a4f4ce2e1fae9d157260`，两个资源槽为 null，FFTS 已开启。设备 kernel 的 ELF 声明参数区为 56 字节。原 launcher 将实际 blockNum 限定为 40，仍向 kernel 传入原始逻辑 grid。PR C++ 参数只有逻辑 grid，会用乘积直接设置 blockNum，此适配器仅在真实 grid≤40 时进入计划；超过时使用原 launcher。[实机 driver、compiler、ELF](target-driver-evidence.json)

60 形状共有 22 个 B×L≤40，其中首步用原 launcher 获取二进制；无其他回退时全程可有 21 次计划提交。50 步窗口为 16 次 eligible，#4–6 是 19、28、44，前两步可进入计划。把 `stage.py` 置于已有直接发布梯度的 stage 之后。A/B 均使用同一 overlay、TASK_QUEUE_ENABLE=1、TORCHINDUCTOR_NPU_FAST_LAUNCH=1；单切 `NPU_EAGER_FOLD_PLAN=0/1`。rank000/eager_fold_plan.json 逐步记录实际 shape、逻辑 grid、当前 stream、真实 Hook 与 profiler 状态、所用二进制、进入及成功返回 C++ 的次数。
