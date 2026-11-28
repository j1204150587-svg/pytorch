# PTA 释放队列的索引交接

`ReleaseQueue::WriteToReleaseQueue` 和 `ReadFromReleaseQueue` 各由一个线程执行。补丁用写索引的 release/acquire 发布已构造的槽位，读索引对称交还可复用的槽位。资源释放仍由原线程按 FIFO 顺序执行。参考基线启用原 01、65,536 项缓存和双向 embedding 描述符。两腿载入同一原生库，只切换 `TORCH_NPU_RELEASE_RING_ACQ_REL`。[实测补丁](release-ring.patch) · [ARM 编译、百万项压力测试及原始计数](repro.tgz)

八步审计中，B 发布、消费各 87,845 项，loss 最大绝对差 5.72×10⁻⁶，梯度、norm、最终权重、AdamW 一致。原训练包含 8 次有限值更新、0 次 overflow 跳步。以下两行分别是 A/B 和 B/A 的实际测量，统一按 A→B 展示。步时为预热 10 步后的 50 步无 profiler 同步窗；Free 占比为 Level0 第 4–6 步的 ΣFree/ΣStage。

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|400.426|377.370|+5.76%|454.629|89.394%|426.702|88.744%|+0.73%|
|375.594|386.129|−2.80%|516.206|90.599%|457.162|89.433%|+1.29%|

单位 ms/步。50 步实际处理 136 个样本、2,994 个 token。两序的样本/s 为 6.793→7.208、7.242→7.044，token/s 为 149.541→158.677、159.428→155.078。两序分别节省 23.056、−10.535 ms/步，配对差的样本标准差 23.752 ms/步。两组 Level0 的实际 shape 均为 [1,19]、[2,14]、[2,22]。Stage 为 508.567→480.826、569.772→511.181 ms/步，Computing 为 53.837→53.977、53.424→53.895 ms/步。[逐腿复算](metrics.json) · [原始 CSV、指纹和启动命令](windows-and-commands.tgz)

Level1 第 4、5 步的 ATen 线程顺序、入队数、物理流 44/45/47 的工作及控制序列全部相同。两步各有 15,093、15,094 个 kernel launch，另有 2,121 个异步 memcpy 和 6 个同步 memcpy。producer 至 enqueue 的 p95 为 26.01→28.35、26.42→28.24 µs，consumer 服务 p95 为 36.29→47.25、37.01→48.15 µs。单笔 kernel launch p95 为 7.72→9.39、7.70→9.19 µs。已入队且无 producer 正在入队的相邻 consumer 间隙与 Free 相交 1.273→5.827、0.926→7.896 ms。这两种方向下早窗 Free 均减少，但单笔 ACL 和双序端到端证据不支持采用。12 条腿均通过卡况检查。[队列等待、consumer、ACL 与真实工作序列](physical-comparison.json) · [逐请求 CSV](summary.tgz) · [原始运行路径和 SHA-256](evidence-index.json)
