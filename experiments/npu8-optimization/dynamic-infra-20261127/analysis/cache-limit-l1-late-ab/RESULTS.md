# D01 CANN executor 容量的延后窗口归因

两腿均使用默认 wheel 和 01 分块 overflow、Task Queue Mode 1、动态 eager。B 启动前设 `ACLNN_CACHE_LIMIT=65536`。#12、#13 的实际 `[B,L]` 分别是首次出现的 `[4,17]` 和 `[1,25]`。#12 起默认容量的同一步重复键有大量 miss，扩容保留了一批可复用 executor。维持各步骤真实算子、物理核、SDMA、流内位置和 ACL 提交量后，三类重复请求的 producer 准备与 consumer 服务 p95 均缩短。[A 腿命令和环境](../../evidence/cache-limit-l1-late-ab/01-A/command.json) · [B 腿](../../evidence/cache-limit-l1-late-ab/02-B/command.json) · [实际命中计数](../../evidence/cache-diag/01-A/operators.csv) · [扩容命中计数](../../evidence/cache-diag/02-B/operators.csv)

## 先核对真实提交

全步 ATen 逐线程序列、队列事件数量、各类 ACL 提交次数均相同。对流 44、45、47 以物理流 ID 与流内工作序号作正反两个方向的完整匹配。#12 的 1,053、8、16,160 个工作任务，#13 的 1,053、8、16,159 个工作任务均同类型、同名、同序，双方无缺项。[提交核对](submissions_compare.json) · [逐物理槽位校验](physical_stream_substitutions.json)

按 producer 线程角色及 ATen 调用序号，经 PTA queue correlation、CANN HostToDevice 两端和物理任务连接，核对高频子群每次真实执行的 1:n 关系。下表两个步骤分别独立满足所列每步数量，A/B 每笔的 ACL API 顺序、物理流、流内序号和核名全相同。

| 每步请求 | 请求数 | 每次实际计算提交 | 本组实发核数 |
|---|---:|---|---:|
| worker `aten::div → aclnnDivs` | 1,039 | 1,033 笔各 2 核，6 笔各 1 核 | 2,072 |
| main `aten::mean → aclnnMean` | 1,033 | 每笔 1 核 | 1,033 |
| main `aten::add → aclnnAdd` | 1,030 | 每笔 1 核 | 1,030 |

另有 main Divs 每步 10 笔各 1 核，worker Add 每步 5 笔各 2 核。下述热点分位数只用表内子群。[Divs 逐笔配对](div/matched_calls.csv.gz) · [Mean 逐笔配对](mean/matched_calls.csv.gz) · [Add 逐笔配对](add/matched_calls.csv.gz)

## 查询次数、分位数和长尾

全步可在同线程 ATen 父 scope 中唯一归属的 `GetWorkspaceSize` 从 **7,567→4,276**、**7,566→4,275**。其中所有线程合计的 Divs 两步均为 1,045→8，Mean 均为 1,033→2，Add 为 1,034→4、1,034→3。embedding 反向在两步依然为 1,026→1,026。各算子每个子群的查询计数和 producer、wait、consumer、launch 的 p50、p95、p99 和最大值保留在原表。[全量延迟统计](operator_latency.csv)

| 子群与 p95（µs） | #12 A→B | #13 A→B |
|---|---:|---:|
| Divs，producer / wait | 40.47→17.96 / 2.64→8.47 | 40.54→19.73 / 2.46→19.63 |
| Divs，consumer / 单次 launch | 43.68→22.40 / 7.10→6.16 | 46.26→22.83 / 6.98→6.17 |
| Mean，producer / wait | 55.93→21.22 / 2.64→41.43 | 61.60→19.15 / 3.13→70.84 |
| Mean，consumer / 单次 launch | 25.93→15.77 / 10.65→9.82 | 25.24→19.41 / 9.03→10.34 |
| Add，producer / wait | 31.41→15.11 / 2.65→37.23 | 35.38→13.66 / 3.82→71.70 |
| Add，consumer / 单次 launch | 25.54→11.33 / 8.71→5.45 | 25.91→15.39 / 7.04→8.14 |

worker Divs 的后继工作空档分别从 **132.68→53.42 ms**、**134.34→40.47 ms**。其中与该笔 producer 的 workspace API 相交 **17.32→0.06 ms**、**17.76→0.03 ms**。main Mean 的对应空档是 41.78→17.79 ms、45.99→28.89 ms，main Add 是 40.13→6.04 ms、41.20→9.48 ms。另一方面，worker embedding 反向的对应空档同期增加 84.1 / 97.4 ms，worker copy 增加 19.3 / 24.1 ms，main copy 增加 10.6 / 16.3 ms。可见工作量守恒，提交时机及其他算子的测量耗时发生变化。官方 Level1 Free 最终只缩短 **22.403 / 1.307 ms**，分别为 637.155→614.752 ms 和 635.431→634.124 ms。[空档归因](gap_attribution.csv) · [A 腿原始 CSV](../../evidence/cache-limit-l1-late-ab/01-A/step_trace_time.csv) · [B 腿原始 CSV](../../evidence/cache-limit-l1-late-ab/02-B/step_trace_time.csv)

## 标准 #4–5 窗口与形状复访

标准 #4、#5 的可归属 workspace 数是 4,275→4,274 和 4,277→4,276。Divs 都是 8→8，Mean 都是 2→2，Add 分别是 3→3 和 4→4。各少的一次查询属于 InplaceCopy。此时两腿三类热点尚未形成容量差；早窗 Level1/Level0 观察到的步时差需与后段容量机制分开解释。标准 Level0 仍按实验约定报告第 4–6 步。[标准窗口逐算子对照](../cache-limit-l1-ab/operator_latency.csv) · [标准窗口流序](../cache-limit-l1-ab/physical_stream_substitutions.json)

另外的形状立即复访诊断中，8 组输入形状各首次及复访一次，前、反向 embedding 各自在首访合计 8,211 miss、5 hit，复访为 0 miss、8,216 hit。这份重复形状诊断与 60 个形状互异的正式性能窗独立。[复访的输入序列](../../evidence/repeat/workspace/configs/dynamic_shapes_npu8.json) · [复访按算子明细](../../evidence/repeat/01-A/repeat-op-totals.csv)

在上级 analysis 目录复算。将目录参数 `cache-limit-l1-late-ab` 改为 `cache-limit-l1-ab` 可复算标准步窗。按调用逐笔配对须在一对腿分析完成后执行。

```text
python -B compare_dynamic_legs.py ../evidence/cache-limit-l1-late-ab/01-A ../evidence/cache-limit-l1-late-ab/02-B cache-limit-l1-late-ab
python -B stream_substitutions.py cache-limit-l1-late-ab
python -B compare_operator_calls.py cache-limit-l1-late-ab --aten aten::div --op aclnnDivs --out cache-limit-l1-late-ab/div
python -B compare_operator_calls.py cache-limit-l1-late-ab --aten aten::mean --op aclnnMean --out cache-limit-l1-late-ab/mean
python -B compare_operator_calls.py cache-limit-l1-late-ab --aten aten::add --op aclnnAdd --out cache-limit-l1-late-ab/add
```
