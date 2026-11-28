# 整包三开关的正确性与激活验证

目标为 `codex/hstu-dynamic-eager-infra-v2.10@e3cda6399cb194c813530222478ea018f5aca5c9`。两腿使用同一份该提交构建的原生库。A 将前向 embedding、反向 embedding、满队列等待者唤醒都设为 0，B 全部设为 1。两腿保留相同的 65,536 项 ACLNN cache、Task Queue 1 和 01 分块 overflow。原八步 HSTU 审计使用实际 worker，独立探针另起冷进程分别验证 CPU 对照和非有限输入。

|项目|大缓存审计 A|大缓存审计 B|
|---|---|---|
|NPU_DYNAMIC_REBASE_INDICES|0|1|
|NPU_DYNAMIC_REBASE_FORWARD|0|1|
|TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY|0|1|
|NPU_DYNAMIC_REBASE_COUNT、TORCH_NPU_QUEUE_WAKE_AUDIT|均 1|均 1|
|ACLNN_CACHE_LIMIT|65536|65536|
|LD_PRELOAD|counter_embedding.so|同 A|
|INFRA_COUNTER_DIR|当前腿的实际输出目录|当前腿的实际输出目录|

只在审计进程启用的计数收集器 SHA-256 为 `5528151fd5388941af47b417b50b7ced00d2dc661d93425d6008d4d53fb228a7`。逐一匹配 worker 的容器 PID、映射库、实际环境、各方向的命中、回退、缓存键和描述符计数，以及该进程退出时的 queue 计数。实际大缓存训练通过原八步 loss 容差、梯度、norm、AdamW 和权重更新门才接受。若满队列没有登记等待者，开启策略应将针对 producer 的无等待者写入降至零。释放路径单独计数。

独立 embedding 微门交错两个底层 storage，覆盖三个实际尺寸、重复尺寸、非零 offset、八个表位置、两个 padding 取值，64 组前向和 64 组反向逐位对照 CPU。降一维的 indices 额外检查每方向一次回退。独立 BF16 ZeRO-2 微门交错 4 个有限输入和 +Inf、NaN、−Inf，比较两腿每步 loss、梯度 SHA、FP32 主权重、BF16 权重、AdamW 状态及实际跳步。每个 `operator_probe.py` 卡锁预约内顺序执行两个互相独立的冷进程。[独立探针入口](probe.py) · [双腿调度](run_gates.py) · [验证程序](verify.py)

在宿主研究根目录执行以下命令。`BRANCH_PACKAGE` 和 `BRANCH_LIB_SHA` 分别取本轮 build/ready.json 的 inner_package 与 sha256。`AUDIT_PAIR` 指向当前 fixed-cache 八步训练收据。

```bash
python3 -B branch-stability-v210/semantics/run_gates.py --package "$BRANCH_PACKAGE" --sha256 "$BRANCH_LIB_SHA" --dest "$PWD/branch-stability-v210/semantics/probes"
python3 -B branch-stability-v210/semantics/archive.py --probes "$PWD/branch-stability-v210/semantics/probes" --audit "$AUDIT_PAIR" --sha256 "$BRANCH_LIB_SHA"
```

## 实机结果

两条原训练腿加载同一原生库，SHA-256 为 `175bd91575354569a63d8599a501b0a19a2a224ada114e44a763ba38fa817d75`。八步 A 的两个方向计数均为零；B 前向命中 8,192 次、回退 24 次、写键 16,384 次、构造描述符 286 次，反向分别为 8,192、24、8,192、128。A/B 均消费 87,845 项，满环登记为零。A 对 producer 执行的 87,066 次普通 eventfd 写入均无登记等待者，B 的该项为零。两腿八步梯度、norm、最终权重和 AdamW 一致，loss 最大绝对差 7.6294×10⁻⁶。两腿均完成八次有限更新；01 的八次 overflow 扫描均走 fast 路径，未回退。[原训练与独立诊断完整收据](activation.json)

独立 embedding 门两腿各验证 64 组前向、64 组反向逐位与 CPU 相等。两种存储所有者、不同表偏移和重复 shape 覆盖了缓存重用。B 的前/反向分别实际命中 64/64 次、各回退 1 次、产生键 128/64 次、构造真实描述符 18/18 次。独立 ZeRO-2 门两腿七步的逐步 loss、梯度哈希、FP32 主权重、BF16 权重、AdamW 状态完全相同；4 次有限更新、3 次非有限输入跳步，三个跳步后的下一次有限输入均继续更新。B 的两个 embedding 方向分别实际命中 7 次。

两个锁预约内分别有 5/4 个整芯片进程快照，所有已采样的占卡 PID 均匹配本腿。四个独立进程的宿主 PID 与容器 PID 均完成对应；芯片定期采样捕获了两腿的 embedding 进程，两个短时 overflow 进程均处在采样间隙。八步训练和独立探针只覆盖列出的合成尺寸及输入。独立注入通过的结论限于这份分支构建和已测类型。[命令、逐步状态、原始计数与整卡收据](evidence.tar.gz)
