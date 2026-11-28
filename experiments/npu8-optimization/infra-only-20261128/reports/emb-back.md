# 大缓存下增加 embedding 反向描述符规范化

第一行为正序 A/B，第二行为反序 B/A，表内均按 A→B 展示。时间单位为 ms/步。

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|501.934|389.568|+22.39%|633.000|92.62%|531.869|91.16%|+1.58%|
|470.964|417.121|+11.43%|600.126|91.97%|488.638|90.49%|+1.61%|

|顺序|腿|样本/s|token/s|Level0 Free ms/步|Stage ms/步|Computing ms/步|
|---|---|---:|---:|---:|---:|---:|
|AB|A|5.419|119.299|633.000|683.411|50.315|
|AB|B|6.982|153.709|531.869|583.451|51.471|
|BA|B|6.521|143.555|488.638|540.001|51.218|
|BA|A|5.775|127.143|600.126|652.553|52.332|

两序 A−B 分别为 +112.366、+53.843 ms/步，两个配对差的样本标准差为 41.382 ms/步。每腿无 profiler 窗口共 136 个样本、2,994 个 token。Level0 三步依次为 `[1,19]`、`[2,14]`、`[2,22]`。Free 占比用三步的 ΣFree/ΣStage。

该项开关为 `NPU_DYNAMIC_REBASE_INDICES`，原生库、额外前置及分母保存在复算数据中。[未舍入数据](emb-back.json) · [AB 无 profiler](../evidence/raw/20261128T042001Z-ab-no-profiler-2nz7v_m8/status.json) · [AB Level0](../evidence/raw/20261128T042358Z-ab-l0-lv1ca44q/status.json) · [BA 无 profiler](../evidence/raw/20261128T042200Z-ba-no-profiler-lqfp6ic5/status.json) · [BA Level0](../evidence/raw/20261128T042603Z-ba-l0-n8a7ve38/status.json)
