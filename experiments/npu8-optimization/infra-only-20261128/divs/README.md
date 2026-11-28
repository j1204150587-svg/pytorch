# 动态 eager 侧表均值反向的标量除法

## 原始负载中的频次和空档

原始 01 训练每步遍历 1,024 张侧表。每张表在 `Embedding` 后沿长度为 4 的特征轴做 `mean(dim=-2)`。在反向中，均值梯度先扩展，再除以 4。原始 Level1 的 `aten::div → Enqueue@aclnnDivs → Dequeue@aclnnDivs → Node@launch` 链对应两项设备工作。按 profiler step、物理流、设备任务 ID 和队列 ID 联接后，恰有 1,024 组满足 BF16 `[B,L,1,64] → [B,L,4,64]` 的 `BroadcastTo`，随后执行 `RealDiv`。另外 9 次三维广播的末轴扩展到 64，候选沿原路径处理。

| 原始 01 Level1 步 | Shape | 本次出现 | 请求 / 实际 launch | BroadcastTo 核 ms | RealDiv 核 ms | BroadcastTo 前全流空档 ms | RealDiv 前全流空档 ms | consumer p50 / p95 / p99 / max µs |
|---|---|---|---:|---:|---:|---:|---:|---|
| 4 | 1 × 19 | 首见 | 1,024 / 2,048 | 5.922 | 1.937 | 85.750 | 0.437 | 15.080 / 17.407 / 21.149 / 55.380 |
| 5 | 2 × 14 | 首见 | 1,024 / 2,048 | 4.656 | 2.223 | 84.535 | 0.713 | 18.925 / 22.627 / 26.318 / 76.400 |
| 12 | 1 × 19 | 再见 | 1,024 / 2,048 | 5.184 | 1.929 | 100.287 | 0.861 | 17.990 / 28.878 / 43.243 / 471.560 |
| 13 | 2 × 14 | 再见 | 1,024 / 2,048 | 4.109 | 2.220 | 111.924 | 0.347 | 17.910 / 24.600 / 36.459 / 142.500 |

每个原始步骤的提交都落在物理流 47。空档按下一个设备工作归组，整组空档涵盖当前调用出现前的等待，不能直接折算收益。第 4 步，以 1,024 组相关空档合计 86.187 ms 为例，24.641 ms 在最内层 `aten::div` 调用前，producer 准备余项 16.654 ms，enqueue 8.708 ms，可见队列等待 2.449 ms，consumer 到 ACL API 前 3.380 ms，ACL 提交 5.679 ms，API 结束后 24.673 ms。workspace 的可见交集仅 0.003 ms。完整分位数及另两类投影矩阵乘对照见 [stock.json](stock.json)。在本目录运行 `python analyze_stock.py` 可复算。

## 更改和证伪条件

目标为 op-plugin `eaf6b55812a1998a4a40e485e3b2861a40e9c2f6` 的 `op_plugin/ops/opapi/DivKernelNpuOpApi.cpp`。只选择无 rounding 的 functional `div(Tensor, Scalar)` 或 `div(Tensor, Tensor)` 的 CPU 单元素标量分支。核对 BF16、除数 4、B=1–4、L=12–32、末两轴大小 [4,64]、实际零 stride、base format、Mode 1 和 eager 流。新的 C++ 路径用一次 `as_strided` 取得保留 storage offset 的紧凑视图，与相同设备和流上的 BF16 常量 `[1,1,4,1]` 做 `aclnnDiv`，把输出直接写到原接口分配的新存储。物理工作预期从每张表两项降为一项。原生计数器顺序为标量入口命中、Tensor 入口命中、功能性标量分支回退和常量初始化。

在相同物理流检查真实 launch 数，保持每步侧表请求为 1,024。若输入/输出或反向结果差异、流或存储别名不符、训练路径零命中，停止正式测量。提交数变化应以原生训练的 Level1 确认。训练同步窗口与 Level0 Free 的收益以各自 A/B、B/A 判定。

## 实现前探针

`probe.py` 用 60 个形状及第二条流的 4 个附加用例比对三种原始调用入口。64 组中，输出 BF16 位、连续布局、非零偏移读取和独立输出均通过。位对比覆盖 ±0、subnormal、±Inf 和带 payload 的 NaN。Python 执行候选表达式时包含一次 `narrow`。

| 同流标记（8 个请求） | ACL launch | 设备任务 | consumer 中位 µs | 各请求 ACL 总时长中位 µs |
|---|---:|---|---:|---:|
| A，第 4 步形状 | 16 | 8 BroadcastTo + 8 RealDiv | 13.210 | 7.725 |
| B，第 4 步形状 | 8 | 8 RealDiv | 10.120 | 5.730 |
| B，第 5 步形状 | 8 | 8 RealDiv | 9.745 | 5.195 |
| A，第 5 步形状 | 16 | 8 BroadcastTo + 8 RealDiv | 13.945 | 8.000 |

无 profiler 的同步微窗口，B−A 的配对中位差在第 4 步形状、第 5 步形状和 60 形状序列上分别为 +7.609、+8.013、+8.186 µs/次。Level1 里 Python `aten::narrow` 含 `slice`、`as_strided`，两形状的均值为 15.821、9.416 µs。这个观测支持试验直接构造视图的原生实现。原始计时和语义见 [micro.json](micro.json)，在本目录运行 `python analyze_probe.py` 可从 [divs-trace.json.gz](divs-trace.json.gz) 复算 [micro-attribution.json](micro-attribution.json)。

## 构建和开关

编译在 `recsys-gr-deepspeed:host88-ds0.19.0-torch2.10.0` 镜像中完成。构建容器使用 `runc`、关闭网络，编译前验证无 NPU 设备节点。原始 PTA 为 `38e483ee79ec8a3286517808740a811c0f0ae66c`。基于原对象和工具链只重编 Div 单元后重新链接，产物为 `/artifacts/infra-only-20261128/divs/build/package`，库 SHA256 为 `04a37fbf47039b1f7f06cd9f0190f7a2130f98207e2a74251270aea1f0ca7286`。完整编译和链接参数见 [build-receipt.json](build-receipt.json)。

```bash
source /workspace/env.sh
cd /artifacts/infra-only-20261128/divs/native
python3 -B build.py --study /artifacts/dynamic-infra-20261127 --output /artifacts/infra-only-20261128/divs/build
```

正式实验两腿使用同一产物，仅 B 启用 `NPU_HSTU_COMPACT_DIV=1`。原生门禁位于 [native_gate.py](native_gate.py)，对照采用原 op-plugin `div.out`，覆盖功能性输入、回退形状、共享存储、两个流与梯度。

实机原生门禁通过 197 组功能性与回退对照、4 组梯度对照。标量入口命中 128 次、Tensor 入口命中 260 次、回退 22 次、常量初始化 3 次。两条流、非零 storage offset、非有限值传播和独立输出均符合检查。真实训练 8 步审计中逐步梯度、最终 BF16 权重、FP32 主权重和 AdamW 状态相同，norm 最大差 0，FP32 loss 最大差 0.000003814697265625。两腿各有 8 次更新和 0 次实际 overflow 跳步。[原生门禁](native-gate.json) · [原训练 8 步审计](../evidence/pairs/divs-audit-ab/result.json)

## 原始 01 训练结果

前两组无 profiler 双序结果方向相反，归入端到端未复现。Level0 第 4–6 步的 Free 和 Free 占比在两次配对中都下降。下表两行依次对应 A/B、B/A，均按 A→B 读取，耗时单位为 ms/步。

| 优化前端到端 | 优化后端到端 | 端到端提升比例 | 优化前 Free | 优化前 Free 占比 | 优化后 Free | 优化后 Free 占比 | Free 占比提升比例 |
|---:|---:|---:|---:|---:|---:|---:|---:|
|604.536|553.043|+8.518%|575.744|91.702%|544.683|90.273%|+1.558%|
|546.407|596.957|−9.251%|637.837|92.546%|611.214|91.227%|+1.425%|

实际 50 步固定 136 个样本、2,994 个 token。正序样本/s 4.499→4.918、token/s 99.051→108.274，反序样本/s 4.978→4.556、token/s 109.589→100.309。两次 A−B 配对差为 +51.493 和 −50.550 ms/步，样本标准差 72.155 ms/步。[同步窗口、Level0 明细和八列复算](formal-pairs.json)

正式 Level1 的 [1,19]、[2,14] 两步各按 autograd worker 的 `aten::div` 调用序号匹配 1,024 笔。目标 ACL launch 均为 2,048→1,024，物理流 47 的当步总设备工作分别为 16,159→15,135 和 16,160→15,136。流 44 和 45 分别保持 1,053 和 8 项。将一个目标侧表除法视作同一操作后，两腿全部流的其余实际任务名、类型、顺序逐项一致。第 4/5 步的 producer 准备 p50 分别为 12.36→20.03、12.49→20.47 µs，consumer 服务 p50 为 14.95→13.91、15.28→14.245 µs，每笔请求累计 ACL launch API 的 p50 为 8.17→6.17、8.10→6.26 µs。该算子后继任务前的全流空档从 60.565→53.962、56.652→58.258 ms，两步方向相反。[逐步与分位数](training-l1/verified.json) · [逐请求明细](training-l1/per_request.csv.gz) · [核验脚本](verify_training.py)
