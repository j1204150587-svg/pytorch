# 前向 Addmm 描述符

第一行为正序 A/B，第二行为反序 B/A，表内均按 A→B 展示。时间单位为 ms/步。

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|581.109|602.520|-3.68%|661.313|92.78%|573.566|91.78%|+1.08%|
|548.188|555.297|-1.30%|641.484|92.58%|558.503|91.56%|+1.10%|

|顺序|腿|样本/s|token/s|Level0 Free ms/步|Stage ms/步|Computing ms/步|
|---|---|---:|---:|---:|---:|---:|
|AB|A|4.681|103.044|661.313|712.789|51.335|
|AB|B|4.514|99.383|573.566|624.966|51.282|
|BA|B|4.898|107.834|558.503|609.973|51.340|
|BA|A|4.962|109.233|641.484|692.894|51.241|

两序 A−B 分别为 -21.411、-7.109 ms/步，两个配对差的样本标准差为 10.113 ms/步。每腿无 profiler 窗口共 136 个样本、2,994 个 token。Level0 三步依次为 `[1,19]`、`[2,14]`、`[2,22]`。Free 占比用三步的 ΣFree/ΣStage。

该项开关为 `NPU_DYNAMIC_ADDMM_REBASE_128`，原生库、额外前置及分母保存在复算数据中。[未舍入数据](addmm.json) · [AB 无 profiler](../evidence/raw/20261128T032353Z-ab-no-profiler-l6ls__1u/status.json) · [AB Level0](../evidence/raw/20261128T034317Z-ab-l0-zlfxdh6o/status.json) · [BA 无 profiler](../evidence/raw/20261128T033636Z-ba-no-profiler-rv1sf2e1/status.json) · [BA Level0](../evidence/raw/20261128T040504Z-ba-l0-_qnr5vc0/status.json)
