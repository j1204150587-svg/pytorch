# 同步窗口和标准 Level0 复算

## 标准 Level0 的逐步时间

下表每组三列依次为 Free / Stage / Computing，单位 ms。各步取自官方 step_trace_time.csv，占比按各腿 ΣFree/ΣStage 计算。

| 轮次与顺序 | 步号与实际 [B,L] | A 的 Free / Stage / Computing | B 的 Free / Stage / Computing |
|---|---|---:|---:|
| r1 · A/B | #4 [1,19] | 51.069 / 55.634 / 4.491 | 45.990 / 49.268 / 3.211 |
| r1 · A/B | #5 [2,14] | 52.042 / 56.742 / 4.620 | 45.670 / 49.068 / 3.343 |
| r1 · A/B | #6 [2,22] | 48.120 / 53.227 / 5.043 | 42.670 / 46.483 / 3.761 |
| r1 · A/B | 三步均值 | 50.410 / 55.201 / 4.718 | 44.777 / 48.273 / 3.438 |

CPU ProfilerStep 墙时和设备 Stage 的边界差逐步列入明细。[含 CPU 边界的逐步 CSV](profile_steps.csv)
