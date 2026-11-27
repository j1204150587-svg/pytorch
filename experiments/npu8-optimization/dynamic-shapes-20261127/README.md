# HSTU 动态形状复测

`train_deepspeed.py` 接受 `--shape-schedule`。JSON 按训练步列出 `[batch_size, seq_len]`，训练循环按当步形状直接生成输入。未指定文件时使用原固定形状。`--seq-len` 设置位置嵌入支持的最大长度。动态序列使用单 rank、累积步数 1，DeepSpeed 配置中的 micro-batch 是初始化值，吞吐按实跑的样本和 token 数计算。

```bash
bash run_training.sh \
  --model-config configs/model_many_tables.json \
  --deepspeed-config configs/deepspeed_zero2_bf16_overflow_on.json \
  --steps 60 --micro-batch-size 2 \
  --shape-schedule configs/dynamic_shapes_npu8.json \
  --steady-warmup-steps 10 --steady-measure-steps 50
```

`run_suite.py` 在 host88 使用历史训练 workspace 快照、同一卡锁和 runner 做配对。`extension.patch` 在公共 steady 与 01 patch 之后接入动态形状和各候选。A 是启用 01 的 eager。三个 B 分别启用 07 的 ZeRO-2 单 rank 梯度打包、分块范数、两项叠加。各组 HSTU 前反向均走 eager。

配对两端用 `dynamic_shapes_npu8.json` 的同一前缀。无 profiler 计第 10–59 步，Level0 取第 4–6 步。每项先做 8 步正确性检查，核对每步梯度、范数、loss 及最终的权重和 AdamW 状态。`recompute.py` 从本地归档的 `runs.json` 与各腿原始 `metrics.json` 和 `step_trace_time.csv` 复算。

```bash
python3 run_suite.py --out "$PWD/runs.json"
python3 recompute.py evidence
```
