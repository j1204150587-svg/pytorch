# 大缓存和反向规范化下增加 embedding 前向规范化

第一行为正序 A/B，第二行为反序 B/A，表内均按 A→B 展示。时间单位为 ms/步。

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|427.120|389.618|+8.78%|519.786|90.83%|444.763|89.20%|+1.79%|
|403.248|364.982|+9.49%|497.374|90.65%|441.175|89.12%|+1.69%|

|顺序|腿|样本/s|token/s|Level0 Free ms/步|Stage ms/步|Computing ms/步|
|---|---|---:|---:|---:|---:|---:|
|AB|A|6.368|140.195|519.786|572.258|52.369|
|AB|B|6.981|153.689|444.763|498.600|53.732|
|BA|B|7.452|164.063|441.175|495.035|53.777|
|BA|A|6.745|148.494|497.374|548.647|51.171|

两序 A−B 分别为 +37.502、+38.266 ms/步，两个配对差的样本标准差为 0.540 ms/步。每腿无 profiler 窗口共 136 个样本、2,994 个 token。Level0 三步依次为 `[1,19]`、`[2,14]`、`[2,22]`。Free 占比用三步的 ΣFree/ΣStage。

该项开关为 `NPU_DYNAMIC_REBASE_FORWARD`，原生库、额外前置及分母保存在复算数据中。[未舍入数据](emb-fwd.json) · [AB 无 profiler](../evidence/raw/20261128T043005Z-ab-no-profiler-eby_qzt9/status.json) · [AB Level0](../evidence/raw/20261128T043349Z-ab-l0-j9qkeh59/status.json) · [BA 无 profiler](../evidence/raw/20261128T043159Z-ba-no-profiler-og2cvtv3/status.json) · [BA Level0](../evidence/raw/20261128T043551Z-ba-l0-_vx5re0t/status.json)
