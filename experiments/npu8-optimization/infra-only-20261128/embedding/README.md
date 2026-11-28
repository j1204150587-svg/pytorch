# 动态 eager 的前、反向 embedding 描述符

在 `op_api::embedding_symint` 和 `op_api::embedding_dense_backward` 的 `EXEC_NPU_CMD` 调用前各建立一个作用域。两者共用 `add_param_to_buf(Tensor)` 和 `ConvertType(Tensor)`。Mode 1 的 `EXEC_NPU_CMD_V1` 在 producer 线程上查 executor 缓存，并在未命中时构造 ACL 张量、查询 workspace，随后将执行回调入队。缓存未命中、executor 创建及实际 ACL launch 可在 Level1 中分别检查。[调用点与缓存改动](embedding.patch)

前向仅匹配 BF16 连续 `[64,64]` 权重和三维 int64 `[B,L,4]` 索引的 TensorImpl，反向仅匹配相同布局的索引。每次前移的描述符基址均为 512 字节的整数倍，剩余 offset 与原始 stride 共同定位每个元素。末端补齐不得越过底层存储。设备、布局、stride、存储边界、队列模式和 capture 守卫不满足时保留原路径。两个方向各有专用的 64 位 key 域，cache hit 的地址重绑与 miss 的描述符创建均使用同一归一化结果。线程局部指针指向最内层作用域，析构时恢复前一层，传给消费线程的提交对象不捕获该指针。[守卫与作用域](index_scope.inc)

构建复用已固定 SHA 的 PTA 和 op-plugin 检出、Ninja 编译数据库和目标链接命令，在独立目录重编三份翻译单元。下面的命令在不挂载设备的 CPU 构建容器内执行。

```bash
cd /artifacts/infra-only-20261128/embedding
python3 -B build.py --output /artifacts/infra-only-20261128/embedding/build --jobs 2
```

同包配对采用以下控制。全部腿均启用 01、Task Queue Mode 1 和 `ACLNN_CACHE_LIMIT=65536`。原训练的模型、ZeRO-2、输入与 runner 由宿主包装逐腿核验。

| 配对 | 反向开关 `NPU_DYNAMIC_REBASE_INDICES` | 前向开关 `NPU_DYNAMIC_REBASE_FORWARD` |
|---|---|---|
| 反向的单项效果 | A=0，B=1 | 两腿 0 |
| 反向已开时前向的增量 | 两腿 1 | A=0，B=1 |

原 8 步语义窗单独启用 `NPU_DYNAMIC_REBASE_COUNT=1`，在实际 worker 退出时读取两组四个计数。每个方向应有 8,192 次 eligible、24 次 fallback。前向和反向每个合格调用分别有两次和一次 key 归一化，实际创建的描述符数受 cache hit 影响。正式计时和 profile 关闭诊断计数。独立的 `micro.py` 交替验证两个方向，覆盖重复索引、padding、末表、非零 offset、owner 地址重绑、未见与已见 shape，以及 rank-2 回退。与 CPU 的逐元素结果或两方向的隔离计数任一不符都可证伪实现。[独立探针](micro.py)

Level1 对第 4、5 步分别核对算子、stream、实发工作项，以及 workspace 调用与后继空档。第 4–6 步 Level0 和无 profiler 的 10+50 步同步窗口需要分别做 A/B 与 B/A。两个独立构建的旧数据只能作为定位线索，本组合的增量从上述同包配对获得。

### 2026-11-28 实际激活与数值结果

两组训练共用原生库 SHA-256 `4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c` 和 `ACLNN_CACHE_LIMIT=65536`。八步原数值审计覆盖 [1,13]、[2,28]、[1,24]、[1,18]、[1,19]、[2,14]、[2,22]、[4,21]。两组 A/B 都得到相同的逐步完整梯度哈希、范数、最终 BF16 权重、FP32 主权重和 AdamW 状态。八步均有限并完成权重更新，观察到的 overflow 跳步数为 0。loss 最大绝对差分别为 0.00000572205 和 0.00000381470。

| 比较和训练腿 | 反向 eligible / fallback / key / descriptor | 前向 eligible / fallback / key / descriptor |
|---|---|---|
| 反向开关 A | 0 / 0 / 0 / 0 | 0 / 0 / 0 / 0 |
| 反向开关 B | 8192 / 24 / 8192 / 128 | 0 / 0 / 0 / 0 |
| 前向增量 A | 8192 / 24 / 8192 / 1136 | 0 / 0 / 0 / 0 |
| 前向增量 B | 8192 / 24 / 8192 / 944 | 8192 / 24 / 16384 / 292 |

独立训练进程中，守卫命中数一致时实际构造的 descriptor 数仍然有波动。以上为工作进程退出时的原生计数，workspace 和 ACL 提交另由 Level1 的时间线计算。[反向实机回执](../evidence/pairs/emb-back-audit-ab/result.json) · [前向增量实机回执](../evidence/pairs/emb-fwd-audit-ab/result.json)

两方向同时开启的独立微测交错执行了 64 对前反向调用，全部与 CPU 逐元素相同。正、反向分别有 128、64 次 key 归一化，各构造 18 个 descriptor，随后两方向各有一次 rank-2 回退。微测不计入训练性能窗。
