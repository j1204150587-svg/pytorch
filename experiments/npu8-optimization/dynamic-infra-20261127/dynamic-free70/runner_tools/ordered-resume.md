# 在独占时段之间恢复动态 NPU 8 成对测量

同一条腿的训练、验证和 profiler 时间窗全部由原生 NPU 8 runner 执行。A 调用原有 `--stage baseline`，B 调用 `--stage screen`。两者都在同一宿主文件锁内执行 `run_leg` 及前后卡况检查。训练的 A 或 B 配方通过原注册表传入，对比原始 01 与组合时 A 使用 `workspace-01`，B 使用完整 staged workspace。

在宿主 `dynamic-infra-20261127` 目录启动。首次注册会核对两个 family 的通过语义门、配方和全部原 runner 源码哈希。同一 registry 名记录固定顺序，后续相同命令从首个未完成的预登记腿继续。

```sh
python3 -u -B dynamic-free70/runner_tools/ordered_resume.py --registry urgent28-r1 --families shape-norm-nocast-v1 stock01-to-native-rms-v2
```

默认每个 family 先执行无 profiler 的 A/B、B/A，再执行 Level0 的 A/B、B/A。无 profiler 每腿十步预热、五十步测量，Level0 每腿跑八步，由原 runner 选择第 4–6 步。`--register-only` 可核对输入并预登记，`--modes` 可在新注册表建立时指定模式顺序。

| 文件 | 内容 |
|---|---|
| `ordered-resume/<registry>/plan.json` | 执行前固定的 family、每对的方向、模式、语义门、源码 SHA 和观测间隔 |
| `<pair>/<slot>/try-*/request.json`、`runner.log` | 该次执行唯一标识、原 runner 命令和唯一生成的新会话路径 |
| 同次 `chip.jsonl`、`decision.json` | 逻辑卡 8 对应板 4 芯片 0 的独立进程采样、cgroup 归属及接收或重试原因 |
| `evidence/<family>-<mode>-<order>/pair-provenance.json` | 两腿实际来源、工作目录哈希、容器、训练命令、边界、两腿间隔和尝试次数 |
| `evidence/.../01-A`、`02-B` 等 | 原成功腿的完整文件硬链接，可由现有报告程序直接复算 |

两秒采样一次。若窗内观测到外部进程，即使训练正常退出也保留未采用的原始输出，只重新执行当前腿。采样无效、断档超过八秒、失败原因不明、运行环境漂移或参数改变都会留下需检查的记录。成对发布前再次调用原报告程序校验相同的实际 shape、随机种子、模型、逐项启用条件、实测参数与训练配置。两腿分开取得的报告同时记录间隔秒数。

离线回归使用两份已归档的真实 AB/BA 输出作为 runner 返回，设备占用由明确标记的测试数据提供。常规 family 和端点 family 各核对四对完整结果，各在首对插入一次外占 B 后恢复。测试还核验重启后不重跑已成功腿，以及缺失、无效或过稀样本的拒收。[调度实现](ordered_resume.py) · [回归程序](test_ordered_resume.py)
