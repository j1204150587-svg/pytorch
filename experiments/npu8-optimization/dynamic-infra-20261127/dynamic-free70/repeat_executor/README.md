# 动态 eager 的 ACLNN repeatable executor 实机探针

每个实际形状、固定的视图形状/stride/offset/storage extent 和物理 stream 建立一个 `aclnnEmbeddingDenseBackward` executor。A 每次建三个 ACL 描述符并调用一阶段，再调用二阶段。B 在一阶段之后立即 `aclSetAclOpExecutorRepeatable`，其余调用只对三个 storage 地址调用 `aclSetInputTensorAddr` / `aclSetOutputTensorAddr` 并调用同一二阶段。原生算法、设备任务和核名是否保持由 Level1 对照判定。两腿直接调用同一套 CANN 算子 API，没有改模型、PTA 队列或使用图。

探针覆盖 `[1,19]` 与 `[2,14]`。每个形状用四组不同的梯度、含重复 ID 的 int64 索引及独立的预填充输出，先以 stock PyTorch NPU 和 CPU 的小整数 BF16 梯度交叉核验，再与直接调用 CANN 的输出逐位比对。保留全部 storage 和 workspace 的 owning tensor，退出前同步原 stream，再销毁自持 executor 与描述符。若不能设置 repeatable，会记下真实返回码并发一次单次二阶段，按同一 stream 同步、释放。

| 模式 | 跨请求变化 | 证伪目的 |
|---|---|---|
| `exact` | 完整 stock 描述符保持不变，四次输入与输出均有独立基址 | 确定目标版本、算子及内核计划能否安全复用，是否写入新输出 |
| `raw-family` | 固定 stride，随表号 1/17/33/49 使用原始 storage offset 和完整 storageDims | 只用于 stock，以逐表原生元数据记录对照核名和实际提交 |
| `bounded` | 同样四张表按 512 字节调整基址，保留 residual 和有界的覆盖长度 | 检查一阶段是否能在同一地址族内只调用一次，实际核名是否随描述符改变 |

每次重绑保证 base 至少 32 字节对齐，`bounded` 采用 512 字节移动。Setter 仅支持换地址，不能改元数据。60 个训练 shape 各出现一次时，仅观察到独立存储上 exact 成功不足以证明同一步有可复用机会。训练使用 Mode 1，producer 和 consumer 的线程、stream 及 executor 所有权必须在以后集成时额外核对；单线程微测的 Host 调用时延不应直接折算为 HSTU 性能。

```bash
python build.py --out <fresh-build>
python micro.py --library <fresh-build>/repeat_probe.so --output <exact-A.json> --geometry exact --mode stock
python micro.py --library <fresh-build>/repeat_probe.so --output <exact-B.json> --geometry exact --mode repeat
python micro.py --library <fresh-build>/repeat_probe.so --output <family-A.json> --geometry raw-family --mode stock --profile <level1-A>
python micro.py --library <fresh-build>/repeat_probe.so --output <family-B.json> --geometry bounded --mode repeat --profile <level1-B>
python compare_trace.py <A-trace.json> <B-trace.json> <family-compare.json> --compare-geometries
```

同步检查通过后，用 `--pipelined` 在单一 stream 上连续提交四笔再同步，检测下一次绑定是否污染尚在队列中的输出。两个形状共八个带 marker 的调用。`compare_trace.py` 通过 CANN HostToDevice flow 核对每笔真实设备任务，缺失 flow、缺少 marker 或核名变化均返回未通过。

CANN v9.1.0 的文档排除了内部 H2D/D2D L0 拷贝、部分同地址 ViewCopy 以及内部创建 device tensor 的情况。源码 `OpExecutorImpl::SetRepeatable` 还检查被标记为 unrepeatable、huge-page mem 及各 launcher 的 repeatable 支持。任一 API 拒绝、输出差异、旧输出被覆盖或跨表换核，都有单独结论。[复用接口文档](https://gitcode.com/cann/opbase/blob/v9.1.0/docs/zh/api/nnopbase/aclnn/aclSetAclOpExecutorRepeatable.md) · [实机 ABI 包装](probe.cpp) · [逐笔探针](micro.py)
