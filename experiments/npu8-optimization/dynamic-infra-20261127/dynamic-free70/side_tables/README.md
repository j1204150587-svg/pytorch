# 动态 eager 的侧表批量算子

`MultiTableHSTU.forward` 原来遍历 1,024 个独立的 `Embedding(64,64)`，从输入 `[B,L,1024,4]` 取本表的 4 个 ID，对查表结果取均值，再从第 0 张表开始逐表做 BF16 加法。输出除以 `sqrt(1024)` 后加入主干。候选在这一处调用 eager dispatcher 的 `npu_hstu_experiment::grouped_side_sum`，主干、输入生成和 ZeRO 注册参数沿用原训练脚本。额外的调用入口位于模型层，后端原型和有序 fold 位于算子层。[接入脚本](stage.py) · [复合算子](side_tables_op.py) · [NPU fold](npu_ordered_fold.py)

每步读取现有 1,024 个 BF16 参数，沿第 0 维拼接为 65,536 行的临时张量。把有效的本表行号映射到 `table_index*64+local_id`，一次 dense embedding 后仍按同一表内的 4 个特征做 mean。临时权重的 autograd 边连回每个原有 Parameter，`CatBackward` 按表切回独立稠密梯度。全模型保留 1,052 个参数张量及 tied item/output 权重，64 行的每张侧表保留自己的重复 ID 梯度累加。标准训练用 `randint(0,64)` 产生有效 ID，越界输入不在本轮训练和数值核验范围内。

Triton 的一个逻辑 program 对应一个真实 token 的 64 个特征。grid 取当步真实 `B×L`，kernel 从 `tl.num_programs(0)×64` 得到跨表步幅。对每张表的加法先在 FP32 相加，再舍回 BF16，沿原表序重复。初值取第 0 张表的均值。原 #05/#11 使用 128 对齐要求，[1,19] 等实际宽度在该守卫下会退出。当前实现移除了这一固定宽度依赖。后端是否还会按 shape 新编译，以及 CANN 内部核和 workspace 如何变化，须从实机缓存和 trace 判断。

## 次数和证伪条件

本项预计每步侧表的 eager embedding、mean、dense embedding backward 从各 1,024 笔收束为各 1 笔，1,023 次逐表加法由单个 Triton fold 承载。前向新增 1 次权重 cat、本表索引映射、有效范围保护和可能的布局连续化。表参数注册、反向分发给各叶子、ZeRO 分区及 AdamW 仍会执行。只有实机证实逐流 ACL 和设备任务数量减少，才能量化真实少发的次数。单次 ABI 提交的时长需另取同一算子和实际提交作比较。

此前固定形状的 #11 图内批量 lookup 在同轮 09 基线上把端到端步时从 59.128 降至 27.381 ms，Free 占比却从 16.08% 升至 90.51%。本次必须复核动态 eager 的实际计算和 Free 分母。若本轮 Computing 约 5 ms/步，通信可忽略时，要使 Free/Stage <70%，Free 需少于约 11.67 ms/步。实际阈值以每次 Level0 的 Stage、Computing 和未重叠通信复算。

## 复现与正确性

~~~sh
python -B stage.py <新建的动态训练工作区副本>
python -B micro_gate.py --device npu --out <本次微测输出>.json
~~~

在 runner 计算 staged workspace 哈希之前应用接入脚本，两腿使用相同文件。A 设 `NPU_DYNAMIC_GROUPED_SIDE=0`，B 设为 `1`。如两腿共用 07 梯度打包或分块 norm，在训练结果中列出各开关。每腿写出 `rank000/grouped_side_counts.json`，B 的 `hits` 应等于该腿训练步数。使用特殊 Embedding、模块级局部 forward hook 或不兼容 dtype、设备及布局时走原逐表入口。实验作用域为当前 HSTU 的普通稠密 Embedding 模块和本表有效 ID。

CPU PyTorch 2.9.1 上，1,024 张表在 `[1,19]`、`[2,14]`、`[2,22]`、`[4,31]` 四个形状下，逐表梯度和侧表输出逐位相同。样本重复使用 ID，并覆盖首末行。构造的 BF16 递推 `[256,1,-256]` 得到 0，所有来源均获得单位梯度。完整 HSTU 的 CPU 测试另验证了 3 个形状的 logits、loss 和全体 1,052 个参数梯度，并触发一次带真实模块 hook 的回退。[侧表微测](cpu-micro.json) · [整模型 CPU 检查](cpu-model.json)

NPU 训练需比较同输入下的 loss、所有稠密参数梯度哈希、有限步及 NaN/±Inf 跳步、BF16/FP32 权重和 AdamW 状态。前向 BF16 累加舍入、CANN 聚合重复 ID 的顺序、以及 Triton 在目标平台的行为由 NPU 门验证。正式端到端配对与 Level0、Level1 分开测量。
