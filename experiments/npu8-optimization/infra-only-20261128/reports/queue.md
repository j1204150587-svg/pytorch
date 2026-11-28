# Task Queue 满环等待者唤醒

第一行为正序 A/B，第二行为反序 B/A，表内均按 A→B 展示。时间单位为 ms/步。

|优化前端到端|优化后端到端|端到端提升比例|优化前 Free|优化前 Free 占比|优化后 Free|优化后 Free 占比|Free 占比提升比例|
|---:|---:|---:|---:|---:|---:|---:|---:|
|578.309|569.639|+1.50%|615.817|92.30%|561.133|91.54%|+0.83%|
|604.269|576.804|+4.55%|574.567|91.62%|589.345|92.03%|-0.45%|

|顺序|腿|样本/s|token/s|Level0 Free ms/步|Stage ms/步|Computing ms/步|
|---|---|---:|---:|---:|---:|---:|
|AB|A|4.703|103.543|615.817|667.167|51.214|
|AB|B|4.775|105.119|561.133|612.996|51.741|
|BA|B|4.716|103.813|589.345|640.394|50.942|
|BA|A|4.501|99.095|574.567|627.115|52.435|

两序 A−B 分别为 +8.670、+27.465 ms/步，两个配对差的样本标准差为 13.290 ms/步。每腿无 profiler 窗口共 136 个样本、2,994 个 token。Level0 三步依次为 `[1,19]`、`[2,14]`、`[2,22]`。Free 占比用三步的 ΣFree/ΣStage。

该项开关为 `TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY`，原生库、额外前置及分母保存在复算数据中。[未舍入数据](queue.json) · [AB 无 profiler](../evidence/raw/20261128T040922Z-ab-no-profiler-xrxxryik/status.json) · [AB Level0](../evidence/raw/20261128T041348Z-ab-l0-szr2bxg6/status.json) · [BA 无 profiler](../evidence/raw/20261128T041134Z-ba-no-profiler-vweyczvz/status.json) · [BA Level0](../evidence/raw/20261128T041557Z-ba-l0-nbwanzoe/status.json)
