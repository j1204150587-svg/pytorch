# 动态 eager Free 目标的运行和复算

在现有宿主实验目录 `/data/recsys-gr-ds019-host88/artifacts/dynamic-infra-20261127` 下执行以下命令。目录保留原 `runner/`、`semantic/` 和 `reporting/`。工具放入 `dynamic-free70/runner_tools/`，侧表候选的源文件放入同级的 `dynamic-free70/side_tables/`。各配方使用独立运行索引并调用原卡锁和容器前后检查。基线和候选都保留 01 分块 overflow、TQ Mode 1、原 60 个真实尺寸的合成输入序列。07 梯度打包、分块 norm 和容量为 65,536 的 executor 缓存在这两个示例配方的两腿共同开启。

```sh
set -e
t=dynamic-free70/runner_tools
p=$t/recipes/packed-norm-cache.json
c=$t/recipes/grouped-side-v1.json
python3 -u "$t/run.py" "$p" l1/baseline l0/baseline no-profiler/baseline
python3 -B "$t/report.py" "$p"
python3 -u "$t/run.py" "$c" gate --workspace /data/recsys-gr-ds019-host88/artifacts/dynamic-infra-20261127/20261127T033119Z-baseline-l1-341x7gd3/workspace
python3 -u "$t/run.py" "$c" l1/ab no-profiler/ab no-profiler/ba l0/ab l0/ba
python3 -B "$t/report.py" "$c"
python3 -B "$t/collect.py" "$c"
```

`dynamic-free70/runs/<配方名>/` 内保存固定配方、runner 指纹和运行索引。各实际 session 使用唯一目录，内部保留命令、实际 shape、库及容器信息、前后卡况、训练指标和 profiler 文件。`runs.json` 保留每次返回的退出码和路径。再次执行同一命令会跳过已经成功的配对，有中断时从该配对的第一腿重启。还有进行中的配对时同样可以运行 `report.py` 查看已完成部分。

新配方在 `common` 中设置两腿共有的环境项，在 `A` 与 `B` 中仅改变一个开关。对于经过工作区接入的算子，`stage` 指向包含 `stage.py` 的源目录，`activation` 描述每腿训练产出的 JSON 计数，以 `"$steps"` 表示当前运行步数。训练用相同的源码副本分别运行 A、B，并用实测计数核对分区、norm 和候选命中。语义门对八个形状检查输入、梯度、norm、权重、AdamW 与三次非有限跳步。常规训练每步执行分块 norm，语义门只有五个有限更新步骤执行。

端到端取无 profiler 同步窗口，第 10–59 步累计 136 个样本、2,994 个 token。Level0 使用第 4–6 步的 ΣFree/ΣStage。两序各自的 B 占比均低于 0.70 且两序 E2E 均短于同配置 A，联合目标才为 true。报告另列绝对 Free 是否双序都缩短，以及 Stage、Computing、吞吐与配对波动。独立 baseline 记录仅展示该配置所处的性能水平。

归档程序调用既有 `collect_receipts.py` 保留原始文件与哈希。下载 `dynamic-free70/receipts/*.tgz` 后可使用原 `reporting/unpack_verified.py` 验证解包。按本轮 shape 序列在本地复算示例为 `python -B runner_tools/report.py runner_tools/recipes/grouped-side-v1.json --receipts <解包根目录> --schedule <dynamic_shapes_npu8.json> --output <报告目录>`。
