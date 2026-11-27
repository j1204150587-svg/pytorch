# 动态 eager 的 CANN executor 缓存容量

本项在 Task Queue Mode 1 中只调整 `ACLNN_CACHE_LIMIT`。CANN opbase v9.1.0 的每线程 cache manager 默认容量为 10,000，最大为 10,000,000。该版本 legacy hash 表满后，`OpExecCacheManager::AddOpExecCache` 会拒绝新条目。op-plugin 的 `hit_cache` 将算子、tensor shape、stride、storage offset、底层存储长度、stream 等编入键，命中后直接用返回的 executor 提交第二段。1,024 张侧表的不同视图和逐步变化的 shape 可产生大量键。若重复的小算子键在旧容量下持续 miss，扩大容量可能缩短这些调用的 producer 准备。一次性键本身无法从加大容量受益。[opbase 默认值](https://gitcode.com/cann/opbase/blob/0e247b8c759c48ae42fda6c6f2bd5daab323be0f/include/nnopbase/opdev/op_cache.h#L36) · [插入与容量检查](https://gitcode.com/cann/opbase/blob/0e247b8c759c48ae42fda6c6f2bd5daab323be0f/src/nnopbase/common/utils/op_cache.cpp#L1378) · [op-plugin Mode 1](https://github.com/Ascend/op-plugin/blob/eaf6b55812a1998a4a40e485e3b2861a40e9c2f6/op_plugin/utils/op_api_common.h#L145) · [张量键与地址](https://github.com/Ascend/op-plugin/blob/eaf6b55812a1998a4a40e485e3b2861a40e9c2f6/op_plugin/utils/op_api_common.cpp#L239)

## 诊断

在既有容器内 `cc -O2 -Wall -Wextra -Werror -fPIC -shared cache_audit.c -o cache_audit.so`。沿用同一 60 步动态训练工作区的诊断副本，预先对它执行 `python mark_steps.py /path/to/staged/train_deepspeed.py`。分别为诊断 A、B 建立空目录，给新训练进程传入 `LD_AUDIT`（该 .so 的绝对路径）和 `NPU_CACHE_DIAG_DIR`（对应腿目录绝对路径）。A 使用容器原有设置，B 设置 `ACLNN_CACHE_LIMIT=65536`。诊断不启用 torch profiler。若容器已有 LD_AUDIT，追加实验库并记录实际值。

诊断库用 glibc 的动态链接审计接口记录 CANN 的 `CanUsePTACache` 及 `PTAGetExecCache`。各 API 仍使用原函数的返回值。每个 producer 线程保存查找的时间戳、算子、hash、命中结果与命中时返回的 workspace 字节数。记录中的 `ineligible` 是 CANN 拒绝尝试，`miss` 是实际返回空 executor，零键 miss 单列。训练进程的首个映射快照、开关和 PID 在 fingerprint 文件中。留存库哈希；诊断 TSV 的头部给出实际查找函数的共享库。正式时间、Level0 和 Level1 使用正常工作区和环境，诊断腿的延迟不参与性能对照。

```bash
python summarize_cache.py /path/to/diag-A /path/to/configs/dynamic_shapes_npu8.json
python summarize_cache.py /path/to/diag-B /path/to/configs/dynamic_shapes_npu8.json
```

按实际训练 PID、producer TID、`CLOCK_MONOTONIC` 和闭开步窗口对齐。输出 `operators.csv`、`steps.csv`，其中 `first`、`within_step`、`earlier_step` 分别代表本次首次查找此非零键、该步已有上一笔、最近一次在更早步骤。启动期参与“已见键”判定，步号 -1 单列。跨进程的绝对 hash 受 stream 句柄影响，只比较算子、实际 shape 与对应步号的计数。RSS 和峰值 RSS 以 KiB 记录，整步窗口前后的内存应分别核对。已返回有效 executor 的不同键累计数只给出已观察命中键数。直接确认“插满”需要同一 worker 的 `op cache is full` 日志或匹配版本/符号的容量分支计数。可从同一训练 PID、对应运行时段的 CANN 日志取样和计数。缺失时按“容量切换后的查找变化”报告。

## 证伪

若两腿的重复键 miss 数和 workspace 慢路径频次未下降，放弃该容量值。若只对重复的完整训练 shape 有效，仅报告重复序列诊断。正反序的无 profiler、Level0 与 Level1 分别评估端到端、Free 和逐笔归因，须在相同步及各 stream 对照算子和设备工作序列。若确认提交数或每笔算子内核类型不同，将按实际行为重新归类。
