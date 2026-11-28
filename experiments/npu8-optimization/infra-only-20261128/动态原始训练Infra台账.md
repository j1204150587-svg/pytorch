# 动态 HSTU 原始训练脚本上的 Infra 实验

原始动态训练上，最低的两组配对结果来自大缓存与双向 embedding 描述符规范化。候选腿 Free 占比为 **89.20% / 89.12%**，无 profiler 为 **389.618 / 364.982 ms/步**。70% 目标还差至少 **19.12 个百分点**。正序吞吐为 6.981 样本/s、153.689 token/s，反序为 7.452 样本/s、164.063 token/s。[稳态和 Level0 的逐腿结果](reports/emb-fwd.md)

2026-11-28，逻辑 NPU 8（板 4、chip 0，容器内 npu:0）。按最初动态 shape 配对的 10 个文件逐项核验 SHA-256。训练前反向均走 eager，1,024 张侧表、原梯度表示与分区、原 norm、01 分块 overflow、单 rank、BF16 ZeRO-2、累积 1、seed 1234、Task Queue Mode 1。每个配对调用原动态 runner，宿主只在 docker exec 传入待测原生库及开关。每腿核验工作区文件、训练 worker 加载的 ELF、设备映射与训练容器。两个矩阵乘候选的独立诊断进程通过 so 析构读取原生计数。[原始指纹](runner/reference.json) · [宿主调度](runner/host_pair.py) · [独立分支](https://github.com/j1204150587-svg/pytorch/tree/codex/npu8-infra-only-20261128)

原输入序列为 60 个互异的合成形状，B 为 1–4、L 为 12–32。无 profiler 的前 10 步预热，后 50 步首尾同步，共 136 个样本、2,994 个 token。Level0 取第 4–6 步 `[1,19]`、`[2,14]`、`[2,22]`，Free 占比为 ΣFree/ΣStage。表中降幅均相对当轮 A 腿。[原形状序列](evidence/archived-runner/dynamic_shapes_npu8.json) · [复算脚本](runner/paired_metrics.py)

## 历史的 600 多与 20 多 ms/步

本轮另跑了两次镜像原装库的完全同配置对照。库 SHA-256 为 `6557790a35e6cabf6300e91b8eda3944bc30d8ac834fff48161cf72d0c6e6b39`。同步步时为 **607.116、621.647 ms**，相差 14.531 ms。Level0 的 Free/Stage/Computing 分别为 **661.802/713.624/51.669** 和 **561.193/612.807/51.500 ms/步**，Free 占比 **92.738%、91.577%**。这两条原装腿本身就相差 1.16 个百分点，因此小幅 Free 比例差需结合实际提交与双序步时解释。[原装复算](reports/stock.json)

先前双序直接对照是 635.654→22.186、599.226→21.529 ms/步。两端之间改变了 1,024 张侧表的查找方式、autograd 梯度叶、ZeRO 分区与发布路径，也改变了实际设备提交量。其中，在已有梯度打包、分块 norm 和大缓存的共同前置下，单独对侧表批量计算就出现 283.913→75.896、326.112→81.244 ms/步。端点包含这些训练实现变化。以下原始脚本实验单独计账。[旧配对和配方](https://github.com/j1204150587-svg/pytorch/blob/975a464a2db9283e00a447c1d149df2ec5003068/experiments/npu8-optimization/dynamic-infra-20261127/dynamic-free70/20261128动态HSTU优化台账.md)

## 候选与证伪条件

60 个训练 `[B,L]` 均首次出现。投影矩阵将前两维展为 `M=B×L`，只产生 52 个不同的 M。10 步预热后，50 步里有 42 次遇到新 M、8 次复访旧 M。Level1 的第 4、5 步分别首次使用 M=19、28。同一形状在进程内再次执行的原有诊断采用独立复访序列。[投影与侧表的首见、复访明细](divs/stock.json)

|开关及函数|原始 trace 锚点及每步触发|机制与证伪条件|
|---|---|---|
|op-plugin `addmm`，`NPU_DYNAMIC_ADDMM_REBASE_128`|前向 8 个投影 Addmm，首见 #4/#5 的 workspace 与后继空档交集 0.819/0.853 ms|对齐且在界内的只读 bias、weight 共用规范化描述符。如实际核、流、次数变化，要重新区分减少提交和缩短准备。|
|op-plugin `mm`，`NPU_DYNAMIC_MM_REBASE_128`|反向 16 个 Mm 中有 8 个 dInput 读取静态权重。其首见 workspace 与空档交集 0.546/0.556 ms|只对符合条件的 RHS 做地址重绑。dWeight 应走回退。热形状既有 workspace 次数为 0。|
|op-plugin `div(Tensor, Scalar)`，`NPU_HSTU_COMPACT_DIV`|每步 1,024 个 mean backward 的 BroadcastTo→RealDiv，#4/#5 核前空档 86.187/85.248 ms|零步长轴恢复单元素视图，二元 Div 在输出时广播。核对每次 2→1 的物理设备工作，原始返回布局、别名与梯度。|
|PTA `Repository::Dequeue/Enqueue`，`TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY`|每步约 9,900 次取队列。正常返回的 producer 会将 write-working 清零|分别识别空闲 producer 与因队列满而等待的 producer。要求避免无等待者的 eventfd 写、满队列能唤醒、任务数一致。|
|op-plugin `embedding` / `embedding_dense_backward`|每步各 1,027 次。其中 1,024 个侧表访问不同 storage offset|在 CANN 缓存 65,536 时逐项切换索引及权重描述符，每次同时更新 executor key 与 ACL tensor 描述。交替 owner、形状复访、回退、真实提交和整步双序均单独核对。|

原训练内置审计运行前 8 个形状，比对梯度哈希、norm、FP32 loss、最终 BF16 权重、FP32 主权重及 AdamW 状态。该序列实际发生 8 次更新、0 次 overflow 跳步。保留原训练输入，不把非有限数微测写作训练跳步。[基线调用和时间分解](https://github.com/j1204150587-svg/pytorch/blob/7e923d3ca83f91c54ff3f2c633ce9de15f84634e/experiments/npu8-optimization/dynamic-infra-20261127/analysis/BASELINE.md)

## 实机激活与正确性

- Addmm 在 8 步中命中 64 次、回退 0，生成 128 次缓存键参数和 34 个描述符。Mm 命中 64 次、回退 64 次，生成 64 次缓存键参数和 17 个描述符。两者的原训练梯度、norm、权重及 AdamW 一致，loss 最大绝对差分别为 3.81e−6、7.63e−6。Mm 独立 60 形状微测比较 480 组输出及梯度，在两个流上逐位相同。每批八次 Mm 的 AB、BA 同步微窗分别为 444.59→232.22、413.99→234.74 µs。[Mm 微测](evidence/mm-micro.json) · [原始配对目录](evidence/pairs)
- 紧凑 Div 的独立实包门通过 197 组输出、布局、别名和回退检查，以及 4 组 autograd 梯度比对。8 步训练中 8,192 次命中、200 次回退、1 次常量初始化。两腿梯度、norm、最终权重和 AdamW 一致，loss 最大绝对差 3.81e−6。[原生微测](evidence/divs-native-gate/native-gate.json) · [实机训练](evidence/pairs/divs-audit-ab/result.json)
- 队列两腿都消费 87,845 项，两腿因环满登记的 producer 均为 0。A 向生产者 eventfd 写 86,874 次，B 写 0 次。该数包括初始化和收尾。梯度、norm、最终权重与 AdamW 一致，loss 最大绝对差 3.81e−6。独立 Linux eventfd 压测另验多 producer 满环、EINTR 及清理唤醒。[训练计数](queue/gate-and-counters.json) · [压力测试](queue/cpu-stress.jsonl)

## 已完成的局部归因

Mm 的 8 个 dInput 请求在两步中均逐次匹配到同一算子、物理流、任务位置及单个设备提交。可见 workspace 各由 8 次降为 2 次。producer 准备 p50 从 79.65→15.63、97.52→16.09 µs，与 workspace 重叠的后继空档只少 0.368、0.580 ms。稳态两序分别回退 2.86%、9.45%。[逐笔与长尾](analysis/mm/cohort.json) · [八列对照](reports/mm.md)

保留大缓存和反向规范化，再单独打开前向规范化，前向每步仍有 1,027 个请求、2,051 次真实 launch，可见 workspace 各由 1,026 次降到 18 次。两步 producer 准备 p95 分别从 56.22→21.59、57.47→24.46 µs，consumer 服务从 62.12→21.99、60.46→28.98 µs。单笔 ACL launch p95 分别从 9.36→9.33、6.49→10.38 µs。对应 workspace 与后继空档交集从 17.488→0.229、10.361→0.043 ms。两序端到端分别缩短 8.78%、9.49%。[前向归因](embedding/analysis/emb-fwd/RESULTS.md) · [同轮八列对照](reports/emb-fwd.md)

在共同的大缓存上打开反向 embedding 描述符规范化，Level1 第 4、5 步各 1,027 个反向请求、6,159 次真实 launch，流和任务位置一致。可见 workspace 各由 1,026 次减少到 18 次。与后继全流空档的交集分别从 55.933→0.801、39.234→0.649 ms。producer 准备 p95 从 137.89→18.72、141.88→19.62 µs。流 47 有 1,024 个 Slice 变为 AsStrided，CANN 选择的内核也有变化。两序端到端步时缩短 22.39%、11.43%，Free 时间和占比同向下降。[逐算子四阶段、空档和工作序列](embedding/analysis/emb-back/RESULTS.md) · [八列对照](reports/emb-back.md)

Task Queue 的 eventfd 改动保持第 4、5 步所有流的设备工作数、控制事件和实际提交次序一致。下一项已入队且所有 producer 空闲的相邻 dequeue 间隙，与 Free 的交集分别从 5.302→2.271、6.529→2.880 ms。单笔 ACL launch p95 却从 7.35→7.85、7.39→7.82 µs。端到端两序分别缩短 8.670、27.465 ms，Free 占比变化方向相反。逐段 trace 支持的局部时长小于整步差，两者不宜直接等同。[两序复算](reports/queue.md) · [实现、压力测试及归因](queue/README.md)

Addmm 两步各有 8 个请求、16 次实际提交，逐次算子、流、物理任务位置和核序列均相同。可见 workspace 各从 8 次降到 2 次。producer 准备 p50 分别从 84.69→21.76、92.40→24.87 µs，p95 却从 151.79→226.26、164.92→248.70 µs。该组后继全流空档仅少 0.440、0.354 ms。50 步端到端两序均回退（−3.68%、−1.30%）。[逐步分位数与物理任务核验](analysis/addmm/cohort.json) · [八列对照](reports/addmm.md)

紧凑 Div 在 Level1 两步的同一个 autograd worker 上均匹配到 1,024 次除法。物理流 47 每步少 1,024 个 BroadcastTo，该组实际 ACL launch 从 2,048 降至 1,024。流 44、45 及其余工作次序一致。第 4、5 步的 producer 准备 p50 从 12.36→20.03、12.49→20.47 µs，consumer 服务从 14.95→13.91、15.28→14.25 µs，每个请求内 ACL 总时长从 8.17→6.17、8.10→6.26 µs。该组请求前的全流空档从 60.565→53.962、56.652→58.258 ms。正式 Level0 的 Computing 反而分别增加 6.59、7.42 ms/步，Free 占比的降幅同时包含分母的变化。50 步端到端正序 +8.52%、反序 −9.25%。[逐次匹配及分位数](divs/training-l1/verified.json) · [八列对照](reports/divs.md)


## 有实测收益的优化

每张表第一行是正序 A/B，第二行是反序 B/A。数据统一按 A→B 展示，时间单位为 ms/步。端到端提升为 1−B步时/A步时，末列为 1−B的Free占比/A的Free占比。正值表示改善。

### 大缓存下增加 embedding 反向描述符规范化

11 月 27 日的 D02 补丁在 op-plugin 的 `aclnnEmbeddingDenseBackward` 调用前规范化索引描述符。11 月 28 日将 D02、D05 合入同一实测库，A/B 用补丁提供的 `NPU_DYNAMIC_REBASE_INDICES` 单独切换 D02。两腿均设置 CANN 原有配置 `ACLNN_CACHE_LIMIT=65536`。[实测补丁·反向调用点](https://github.com/j1204150587-svg/pytorch/blob/dd8f8f9751ae128258ab9faa9901f595b880236a/experiments/npu8-optimization/infra-only-20261128/embedding/embedding.patch#L29-L35) · [反向开关与适用条件](https://github.com/j1204150587-svg/pytorch/blob/dd8f8f9751ae128258ab9faa9901f595b880236a/experiments/npu8-optimization/infra-only-20261128/embedding/index_scope.inc#L119-L133) · [D02 原始实现](https://github.com/j1204150587-svg/pytorch/blob/dd8f8f9751ae128258ab9faa9901f595b880236a/experiments/npu8-optimization/dynamic-infra-20261127/rebase/apply_rebase.py)

两腿共用 65,536 项 CANN 缓存并关闭前向规范化，只切换反向索引描述符。 按 A−B 计，两序分别节省 +112.366、+53.843 ms/步。[吞吐、配对波动及原始证据](reports/emb-back.md)

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|501.934|389.568|+22.39%|633.000|92.62%|531.869|91.16%|+1.58%|
|470.964|417.121|+11.43%|600.126|91.97%|488.638|90.49%|+1.61%|

### 大缓存和反向规范化下增加 embedding 前向规范化

11 月 27 日的 D05 补丁在 op-plugin 的 `aclnnEmbedding` 调用前规范化只读权重和索引的描述符。11 月 28 日在同一实测库内，用补丁提供的 `NPU_DYNAMIC_REBASE_FORWARD` 单独切换 D05，测量其在 D02 上的增量。[实测补丁·前向调用点](https://github.com/j1204150587-svg/pytorch/blob/dd8f8f9751ae128258ab9faa9901f595b880236a/experiments/npu8-optimization/infra-only-20261128/embedding/embedding.patch#L11-L17) · [前向开关与适用条件](https://github.com/j1204150587-svg/pytorch/blob/dd8f8f9751ae128258ab9faa9901f595b880236a/experiments/npu8-optimization/infra-only-20261128/embedding/index_scope.inc#L135-L152) · [D05 原始实现](https://github.com/j1204150587-svg/pytorch/blob/dd8f8f9751ae128258ab9faa9901f595b880236a/experiments/npu8-optimization/dynamic-infra-20261127/rebase_forward/apply_forward.py)

两腿共用 65,536 项 CANN 缓存并开启反向规范化，只切换前向只读权重和索引的描述符。 按 A−B 计，两序分别节省 +37.502、+38.266 ms/步。[吞吐、配对波动及原始证据](reports/emb-fwd.md)

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|427.120|389.618|+8.78%|519.786|90.83%|444.763|89.20%|+1.79%|
|403.248|364.982|+9.49%|497.374|90.65%|441.175|89.12%|+1.69%|

### Task Queue 满环等待者唤醒

11 月 28 日新增的 TorchNPU/PTA 补丁提供 `TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY`。`Repository::Enqueue` 登记因队列已满而等待的生产者，`Repository::Dequeue` 在正常出队时仅在有等待者时写入唤醒用的 eventfd。该开关在 Task Queue Mode 1 且关闭 per-stream queue 时生效。[实测补丁·开关定义](https://github.com/j1204150587-svg/pytorch/blob/dd8f8f9751ae128258ab9faa9901f595b880236a/experiments/npu8-optimization/infra-only-20261128/queue/producer-wake.patch#L32-L40) · [入队登记与出队唤醒](https://github.com/j1204150587-svg/pytorch/blob/dd8f8f9751ae128258ab9faa9901f595b880236a/experiments/npu8-optimization/infra-only-20261128/queue/producer-wake.patch#L90-L122)

两腿共用 Mode 1，只切换 producer 满环等待者唤醒。两序端到端均缩短，早窗 Free 占比方向不一致。 按 A−B 计，两序分别节省 +8.670、+27.465 ms/步。[吞吐、配对波动及原始证据](reports/queue.md)

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|578.309|569.639|+1.50%|615.817|92.30%|561.133|91.54%|+0.83%|
|604.269|576.804|+4.55%|574.567|91.62%|589.345|92.03%|-0.45%|

## 尝试无优化路径

首轮三项分别保留已证实的局部改进。把无 profiler 的双序端到端作为是否取得整步收益的判据。

### 侧表反向 Div 广播

两腿使用原装缓存配置。设备工作数与早窗 Free 占比均下降，稳态步时两序方向相反。 按 A−B 计，两序分别节省 +51.493、-50.550 ms/步。[吞吐、配对波动及原始证据](reports/divs.md)

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|604.536|553.043|+8.52%|575.744|91.70%|544.683|90.27%|+1.56%|
|546.407|596.957|-9.25%|637.837|92.55%|611.214|91.23%|+1.42%|

### 前向 Addmm 描述符

两腿使用原装缓存配置，只切换前向投影描述符。两序稳态步时均回退。 按 A−B 计，两序分别节省 -21.411、-7.109 ms/步。[吞吐、配对波动及原始证据](reports/addmm.md)

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|581.109|602.520|-3.68%|661.313|92.78%|573.566|91.78%|+1.08%|
|548.188|555.297|-1.30%|641.484|92.58%|558.503|91.56%|+1.10%|

### 反向 Mm 描述符

两腿使用原装缓存配置，只切换反向 dInput 的权重描述符。两序稳态步时均回退。 按 A−B 计，两序分别节省 -16.923、-53.214 ms/步。[吞吐、配对波动及原始证据](reports/mm.md)

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|592.478|609.401|-2.86%|589.226|91.98%|533.182|91.39%|+0.64%|
|562.936|616.150|-9.45%|641.743|92.39%|624.698|92.21%|+0.19%|

### Divs 复用 producer 侧序列化键

11 月 28 日在 op-plugin 标量 Div 路径测试了 `NPU_DIV_GEOMETRY_KEY`。双向 embedding 和 65,536 项 CANN 缓存在两腿均开启。该补丁对形状、stride、offset、dtype、stream 等条件相同的调用复用先前的 hash，每次仍绑定当前地址并查询 CANN executor。[实测补丁](https://github.com/j1204150587-svg/pytorch/blob/ed8e999aaf6220b19dabf85a1c64a2265d1739bd/experiments/npu8-optimization/infra-only-20261128/div-geometry/from-stock.patch) · [补丁与六个实编文件的 SHA 对照](div-geometry/patch-verification.json)

两次数值门均通过，128 次复用的审计 hash 与原路径逐笔相同。每腿取 9 个同步微窗的中位数。对 `[1,19]`、`[2,14]` 和全 60 形状，B 相对 A 的每次调用在正序分别慢 0.812、0.610、0.833 µs，反序慢 1.772、1.106、2.754 µs。三个负载双序均回退，在算子微测结束，未进入端到端和 Level0 测量。[完整微窗、触发与回退明细](div-geometry/README.md)

原版 !42634 的 eager 入口无命中。前述优化的累计效果若要以百分比归因，需要同轮端点对照。本轮只切换单项，表中增量均使用各自同轮基线。

## 数据边界与归档

本轮有效对照 38 对，共 76 腿。其中 6 对使用原训练的内置数值审计，另 32 对全部经过同步计时或 profiler。后 64 腿每腿都有 7–16 个芯片进程采样，至少 3 个样本实际捕获到训练 worker。有效性能腿均未检出外部作业或采样失败。5 个因占卡失败的尝试及 1 个原装库映射验证失败的尝试均保留在排除清单。12:51（北京时间）收尾时卡上无进程，训练容器仅余 init，公共锁可获取。[逐腿卡况](evidence/slot.json) · [排除清单](evidence/excluded.json) · [收尾快照](evidence/final-slot.json)

数据仅覆盖单 rank、60 个合成形状，batch size 1–4、序列长度 12–32。真实业务频率、长序列与多卡没有在本轮实测。每项训练数值门均使用同一组 8 个有限步，包含 8 次真实权重更新。保留了非有限值算子微测及其适用范围，真实训练 overflow 跳步覆盖数为 0。

板 4、chip 0 出现外部占用时，runner 在前置或后置检查拦截。受干扰的整对已剔除，重新从首腿运行。另有一条原装库诊断因容器 overlayfs 的设备号与映射号不同被剔除，修正宿主的映射校验后整对重跑。正式统计以通过原始工作区、加载库和卡况核对的两腿为准。[复现命令](复现与证据.md) · [原始 CSV、trace 路径和 SHA-256](evidence/原始数据索引.json) · [剩余空档及 70% 差距](剩余空档.md)

本轮实测代码、六组单开关对照、原装对照及复算数据位于独立分支 `codex/npu8-infra-only-20261128`，实测归档提交 `dd8f8f9751ae128258ab9faa9901f595b880236a`。完整原始 profiler 树保存在每条索引记录的宿主目录，本地 evidence/raw 中保留压缩 trace 和原始 CSV。[不可变归档](https://github.com/j1204150587-svg/pytorch/tree/dd8f8f9751ae128258ab9faa9901f595b880236a/experiments/npu8-optimization/infra-only-20261128) · [实验分支](https://github.com/j1204150587-svg/pytorch/tree/codex/npu8-infra-only-20261128)
