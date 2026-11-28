# 动态 HSTU 原始训练脚本上的 Infra 实验

2026-11-28，逻辑 NPU 8（板 4、chip 0，容器内 npu:0）。按最初动态 shape 配对的 10 个文件逐项核验 SHA-256。训练前反向均走 eager，1,024 张侧表、原梯度表示与分区、原 norm、01 分块 overflow、单 rank、BF16 ZeRO-2、累积 1、seed 1234、Task Queue Mode 1。每个配对调用原动态 runner，宿主只在 docker exec 传入待测原生库及开关。每腿核验工作区文件、训练 worker 加载的 ELF、设备映射与训练容器。两个矩阵乘候选的独立诊断进程通过 so 析构读取原生计数。[原始指纹](runner/reference.json) · [宿主调度](runner/host_pair.py) · [独立分支](https://github.com/j1204150587-svg/pytorch/tree/codex/npu8-infra-only-20261128)

原输入序列为 60 个互异的合成形状，B 为 1–4、L 为 12–32。无 profiler 的前 10 步预热，后 50 步首尾同步，共 136 个样本、2,994 个 token。Level0 取第 4–6 步 `[1,19]`、`[2,14]`、`[2,22]`，Free 占比为 ΣFree/ΣStage。表中降幅均相对当轮 A 腿。[原形状序列](evidence/archived-runner/dynamic_shapes_npu8.json) · [复算脚本](runner/paired_metrics.py)

## 历史的 600 多与 20 多 ms/步

先前双序直接对照是 635.654→22.186、599.226→21.529 ms/步。两端之间改变了 1,024 张侧表的查找方式、autograd 梯度叶、ZeRO 分区与发布路径，也改变了实际设备提交量。其中，在已有梯度打包、分块 norm 和大缓存的共同前置下，单独对侧表批量计算就出现 283.913→75.896、326.112→81.244 ms/步。端点包含这些训练实现变化。以下原始脚本实验单独计账。[旧配对和配方](../dynamic-infra-20261127/dynamic-free70/20261128动态HSTU优化台账.md)

## 候选与证伪条件

60 个训练 `[B,L]` 均首次出现。投影矩阵将前两维展为 `M=B×L`，只产生 52 个不同的 M。10 步预热后，50 步里有 42 次遇到新 M、8 次复访旧 M。Level1 的第 4、5 步分别首次使用 M=19、28。同一形状在进程内再次执行的原有诊断采用独立复访序列。[投影与侧表的首见、复访明细](divs/stock.json)

|开关及函数|原始 trace 锚点及每步触发|机制与证伪条件|
|---|---|---|
|op-plugin `addmm`，`NPU_DYNAMIC_ADDMM_REBASE_128`|前向 8 个投影 Addmm，首见 #4/#5 的 workspace 与后继空档交集 0.819/0.853 ms|对齐且在界内的只读 bias、weight 共用规范化描述符；如实际核、流、次数变化，要重新区分减少提交和缩短准备。|
|op-plugin `mm`，`NPU_DYNAMIC_MM_REBASE_128`|反向 16 个 Mm 中有 8 个 dInput 读取静态权重；其首见 workspace 与空档交集 0.546/0.556 ms|只对符合条件的 RHS 做地址重绑；dWeight 应走回退。热形状既有 workspace 次数为 0。|
|op-plugin `div(Tensor, Scalar)`，`NPU_HSTU_COMPACT_DIV`|每步 1,024 个 mean backward 的 BroadcastTo→RealDiv，#4/#5 核前空档 86.187/85.248 ms|零步长轴恢复单元素视图，二元 Div 在输出时广播。核对每次 2→1 的物理设备工作，原始返回布局、别名与梯度。|
|PTA `Repository::Dequeue/Enqueue`，`TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY`|每步约 9,900 次取队列；正常返回的 producer 会将 write-working 清零|分别识别空闲 producer 与因队列满而等待的 producer。要求避免无等待者的 eventfd 写、满队列能唤醒、任务数一致。|
|op-plugin `embedding` / `embedding_dense_backward`|每步各 1,027 次；其中 1,024 个侧表访问不同 storage offset|在 CANN 缓存 65,536 时逐项切换索引及权重描述符，每次同时更新 executor key 与 ACL tensor 描述。交替 owner、形状复访、回退、真实提交和整步双序均单独核对。|

原训练内置审计运行前 8 个形状，比对梯度哈希、norm、FP32 loss、最终 BF16 权重、FP32 主权重及 AdamW 状态。该序列实际发生 8 次更新、0 次 overflow 跳步。保留原训练输入，不把非有限数微测写作训练跳步。[基线调用和时间分解](../dynamic-infra-20261127/analysis/BASELINE.md)

## 实机激活与正确性

- Addmm 在 8 步中命中 64 次、回退 0，生成 128 次缓存键参数和 34 个描述符。Mm 命中 64 次、回退 64 次，生成 64 次缓存键参数和 17 个描述符。两者的原训练梯度、norm、权重及 AdamW 一致，loss 最大绝对差分别为 3.81e−6、7.63e−6。Mm 独立 60 形状微测比较 480 组输出及梯度，在两个流上逐位相同。每批八次 Mm 的 AB、BA 同步微窗分别为 444.59→232.22、413.99→234.74 µs。[Mm 微测](evidence/mm-micro.json) · [原始配对目录](evidence/pairs)
- 紧凑 Div 的独立实包门通过 197 组输出、布局、别名和回退检查，以及 4 组 autograd 梯度比对。8 步训练中 8,192 次命中、200 次回退、1 次常量初始化。两腿梯度、norm、最终权重和 AdamW 一致，loss 最大绝对差 3.81e−6。[原生微测](evidence/divs-native-gate/native-gate.json) · [实机训练](evidence/pairs/divs-audit-ab/result.json)
- 队列两腿都消费 87,845 项，两腿因环满登记的 producer 均为 0。A 向生产者 eventfd 写 86,874 次，B 写 0 次。该数包括初始化和收尾。梯度、norm、最终权重与 AdamW 一致，loss 最大绝对差 3.81e−6。独立 Linux eventfd 压测另验多 producer 满环、EINTR 及清理唤醒。[训练计数](queue/gate-and-counters.json) · [压力测试](queue/cpu-stress.jsonl)

## 已完成的局部归因

紧凑 Div 在 Level1 两步的同一个 autograd worker 上均匹配到 1,024 次除法。物理流 47 每步少 1,024 个 BroadcastTo，该组实际 ACL launch 从 2,048 降至 1,024。流 44、45 及其余工作次序一致。第 4、5 步的 producer 准备 p50 从 12.36→20.03、12.49→20.47 µs，consumer 服务从 14.95→13.91、15.28→14.25 µs，每个请求内 ACL 总时长从 8.17→6.17、8.10→6.26 µs。该组请求前的全流空档从 60.565→53.962、56.652→58.258 ms。正式 Level0 的 Computing 反而分别增加 6.59、7.42 ms/步，Free 占比的降幅同时包含分母的变化。50 步端到端正序 +8.52%、反序 −9.25%。[逐次匹配及分位数](divs/training-l1/verified.json) · [八列对照](reports/divs.md)

