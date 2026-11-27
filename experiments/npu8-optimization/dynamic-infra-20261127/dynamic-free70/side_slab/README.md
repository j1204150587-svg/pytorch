# 动态 eager 侧表的连续权重和单次稠密梯度

本项接在 grouped-side v1 和单 rank 07 梯度分区后。只切换 `NPU_DYNAMIC_SIDE_SLAB`。A/B 共用 07、相同的真实尺寸输入和按表顺序的 BF16 fold。该入口服务于单次前向、一次完整 loss.backward、一次分区和一次 engine.step 的 HSTU 训练契约。业务模型仍注册 1,024 个独立侧表 Parameter，原 checkpoint 键、AdamW 分区和梯度注入位置沿用原训练。[实现](side_slab.py) · [在 v1 之上接入](stage.py)

**原始瓶颈。** v1 每步对 1,024 个权重做一次 `cat`，前向批量 lookup 和反向 dense embedding 已合并。第 4 步 Level0 的 1,052 个叶子 AccumulateGrad evaluate 合计 24.57 ms，5 个 CatBackward 合计 21.48 ms，其中 1,036 个 narrow 合计 17.24 ms。父子 scope 的计时存在包含关系。

**实现。** 07 初始化后检查一个没有 padding 的 BF16 ZeRO 分区、正确的 dtype 和基础张量布局、1,024 个独立叶子及其在参数组和底层 flat storage 中的确切次序、每张表的地址和可达区间。为整段一次创建 `[65536,64]` 的 detached 叶子。该叶子与已有 ZeRO BF16 权重共享存储且由普通 Python 状态对象持有，模型、DeepSpeed 和 AdamW 的参数列表不新增叶子。动态前向继续使用 v1 实测的批量 F.embedding、mean 和有序 fold，直接读取 ZeRO 连续权重；本次 autograd 只为整段侧表生成一个真实的 dense dW。

**逐步发布。** 状态对象预建独立的 8 MiB BF16 梯度缓冲和 1,024 个不重叠的表视图。进入原 07 epilogue 前，把该步 dW 一次复制进缓冲，将预建视图交给原 Parameter.grad。07 原先的分区、裁剪、overflow 与更新读取这些真实梯度。随后清空 slab 叶子的 grad，避免跨步累加。纳入端到端计时的额外工作包括这一次 8 MiB D2D 和 Python 的 grad 指针发布。权重 cat、逐叶 CatBackward/AccumulateGrad 及每步重建梯度视图是实际减少的目标。最终的 kernel、SDMA 和 CPU scope 次数以 A/B trace 为准。

**生效和回退。** 启动期分别记录 prerequisites、模型布局、参数分布和 hooks 条件。每步只在分区、同一 flat 身份和基址、表的身份/地址、训练状态和梯度生命周期仍满足受控 runner 契约时启用。未启用该项时继续运行 v1 分支。训练期间可观测的参数 hook、冻结、非空原 grad、参数替换、地址变更等支持回退。已经开始而尚未发布的 forward 再次进入、缺少应有的稠密 dW、混入独立原参数梯度时中止该腿，以免重复或覆盖更新。微测的运行期 hook 回退核验了原表参数能收到真实梯度。该实验不用于绕过单独读取各表梯度的 autograd.grad API、保存并重入计算图、模型动态重定义或异步改写参数布局。

~~~sh
python -B ../side_tables/stage.py <新建的训练工作区>
python -B stage.py <同一个训练工作区>
~~~

B 的 `side_slab_counts.json` 应包含一次 `installed`、1,024 个 `views_created`，每步各一次 `slab_forwards`、`dense_gradients` 和 `gradient_buffer_copies`，每步 1,024 个 `published_table_views`。新旧权重存储、分区钩子和现有梯度都在守卫范围内。注入门在 07 发布后修改实际分区梯度，使 overflow 跳步和有限恢复经过同一优化器。

CPU PyTorch 2.9.1 模拟单 rank flat，在四个实际 shape 下验证全部 4,380,544 个梯度元素、logits 和 BF16 更新后权重逐位相同，覆盖首末行、重复 ID、一次注入 Inf 的跳步、一步运行期原参数 hook 的回退。真实 DeepSpeed、FP32 master、AdamW、NPU kernel 和整步收益均以实机腿为准。[CPU 明细](cpu-micro.json) · [微测](micro_gate.py)
