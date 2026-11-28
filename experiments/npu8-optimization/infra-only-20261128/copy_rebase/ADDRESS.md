# 动态 HSTU 的 D2D 地址与提交就绪时间

原 runner 的 8 步语义审计使用双向 embedding 描述符规范化、01 overflow、Task Queue Mode 1、1,024 张侧表。两腿运行同一库，B 仅增加地址等价身份的观测。每个操作都从实际张量地址和实际字节长度计算左右边界，跨源、宿及相邻操作复用同一匿名地址编号。统计包含调用顺序、物理流号、dtype、格式、完整尺寸与步长的相等性，以及 StorageImpl、TensorImpl 的同一性。[原始事件](address-evidence/copy-address-344086.tsv) · [worker 身份](address-evidence/worker.json) · [原语义审计](address-evidence/result.json)

| 观测 | 流 44 | 流 47 |
|---|---:|---:|
| 8 步原生 copy 入口调用 | 8,424 | 17,314 |
| 不同 TensorImpl 的同址、同长度、同几何区间 | 8 | 0 |
| 同 dtype、连续、同基格式段内双端同向邻接 | 每步 1,052 个非同址 copy 的 1,051 条边全部邻接 | 8,376 条可比边中 0 条 |
| 只在目标端邻接 | 每步上述 1,051 条边均在两端邻接 | 8,376 条 |

流 44 的每一组从一个 8,761,088 字节的同址 BF16 区间开始，随后是 1,052 笔非同址 copy。8 个组在相同位置出现，后续区间在源和目标两端按地址递增次序拼接。[按步复算数据](address-evidence/summary.json) · [复算程序](address_report.py)

双 embedding 基准的独立 Level1 在第 4 步 [1,19]、第 5 步 [2,14] 各观察到唯一一次没有物理任务的 InplaceCopy，两次都是主线程本步第 10 个 InplaceCopy，紧随其后的 1,052 个主线程 InplaceCopy 都向流 44 各提交一次 SDMA。空请求的 queue ID 分别为 54691、64614，ACL launch 为零，workspace 准备耗时 8.10、8.18 µs，最内层 ATen copy 耗时 29.93、30.11 µs。跨实验的一一复现顺序和数量支持大同址区间已经省去设备复制。独立审计未将原始地址写入同一 Level1 时间线。[Level1 对应项](address-evidence/self-level1.json) · [复算程序](self_trace.py)

再按同一 consumer 的真实 dequeue 顺序检查后续请求 enqueue 完成时间。第 4 步只有 3 组三项在首项开始 dequeue 时都已发布，乐观上限为 6/1,052 次可省提交，第 5 步为 0/1,052。地址数据与时间线分开采集，未逐笔测得二者的交集。若等待将来才 enqueue 的请求，需要另行证明同步边界、发布依赖及增加的排队延迟。当前数据不足以支持在原始逐次 eager 语义下合并整段。[Level1 分组复算](../embedding/analysis/emb-fwd/residual.json)
