# 动态侧表 ID 的表序打包

在 native RMS v2 的 B 腿上追踪 `sum_from_weight` 从 `aten::movedim` 到下一次 `aten::embedding` 的输入。第 4、5 步均在物理流 47 依次发出 Ge、Lt、bitwise_and、add、fill、where 六笔请求。Ge、Lt、add 各多触发一次转置，共九个实际工作。两步的 Host 区间为 366.16、359.55 μs，与 profiler Free 的交集为 0.274、0.258 ms；九个工作首尾跨度为 0.231、0.247 ms，包括 0.104、0.114 ms 设备工作以及 0.127、0.132 ms 内部全流空档。首个 Ge 前另有 2.029、2.098 ms 的空档，其起点早于 ID 准备，不能用它推算融合收益。[任务 ID、物理流、请求和各段时延](stock-scope.json) · [原始 trace 复算](stock_scope.py)

独立校时的第二份 Level1 中，同一边界的 Host 区间为 405.24、389.01 μs，六个请求仍对应九个设备工作，与官方 Free 的交集为 0.312、0.290 ms。首个 Ge 的转置前全流空档为 2.239、2.358 ms，其中 2.086–2.094、2.192–2.200 ms 已发生在对应 `aten::ge` 入口以前。九笔工作内部的全流空档分别为 0.160、0.135 ms。[逐笔校时结果](clock-stock-scope.json) · [共同时钟的物理约束](../diagnostics28/clock-analysis/RESULTS.md)

生成的 `global_ids[T,B,L,4]` 使用真实 `offsets[t]`。以一个动态 `(B×L,16,1)` 网格处理 1,024 张表，每个逻辑块覆盖 64 张表的四个 feature；每表 64 行，非法局部 ID（含负数及 int64 极值）填入原 sentinel 65,536。有效值与 offset 按 64 位模加法。输入切片、非连续布局、不支持的表数/行数、格式、设备或对齐走原 ATen，权重维度另在训练接入验证。输入和结果在实际消费流上记录分配器生命周期。S 为普通 Triton，P 对同一已加载二进制、签名、driver 代码及 grid cap 校验后使用真实 PR C++，首次普通提交只执行一次。[Triton 核](side_ids_triton.py) · [动态守卫和当前 stream](side_ids_impl.py) · [训练接入](stage.py)

独立 CPU 对照在 60 个实际 shape、两种整数类型上验证 120 组地址映射，覆盖任意表偏移、极值、无效和重复 ID。CPU 结果仅说明逻辑。"A/S/P" 微门在上卡后直接比较三路张量，检查跨流早释放、四类回退，再分别对含分配的完整调用与预分配后的同核提交做同步窗口；Level1 对 A/S/P 分别标记并验证每步设备工作数、名称及流。[CPU 映射和结果](cpu-mapping.json) · [实机微测](probe.py) · [Level1 核对](profile_check.py)

训练沿用原生 RMS v2、直接发布和同版本的 TorchNPU 隔离包。A→S 只切索引融合，S→P 只切同核的 Host 提交。复测函数分别记录 `stock`、`triton`、`cold_triton`、`cpp_returned` 和回退原因。若位值、输入生命周期、处理腿真实命中或可辨认的设备工作不相符，终止对应结论。[融合配方](side-ids-fused-v1.json) · [同核提交配方](side-ids-plan-v1.json)
