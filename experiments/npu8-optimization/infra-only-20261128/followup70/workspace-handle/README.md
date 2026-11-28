# embedding 热命中时的 workspace 句柄

正序稳态步时增加 0.48%，反序增加 3.09%。Free 占比正序增加 0.544 个百分点，反序下降 0.842 个百分点。该项归入「尝试无优化路径」。[双序结果、实际吞吐及配对波动](双序结果.md)

两腿在原动态 eager 训练上共用双向 embedding 描述符补丁和 65,536 项缓存。B 在 op-plugin 的 embedding 前向与反向热缓存路径中，向同一 NPU 缓存分配器取得 DataPtr 持有 workspace，省去临时 Tensor 和 Storage 的构造。8 步原训练两腿各请求 16,118 次非零 workspace 和 5 次零 workspace，申请字节数均为 135,832,661,504。这是累计请求量，同一块内存可被重复使用。[实际补丁](workspace-handle.patch) · [原生计数](counts.json)

原训练八步梯度、权重与 AdamW 一致，loss 最大绝对差为 3.81×10⁻⁶。独立 ZeRO-2 探针另行比对四次有限更新与三次非有限跳步。Level1 第 4、5 步对应形状 [1,19] 和 [2,14]，两腿在各物理流的设备工作序列相同，全步 kernel launch 均为 15,093 和 15,094 次。第 4 步全体 enqueue 分类计数有差异，逐流提交与 ACL API 计数已单独比对。[按步核对](physical-comparison.json) · [探针、构建和运行收据](repro.tgz)
