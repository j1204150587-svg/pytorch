# TorchNPU Mode 1 消费线程复用 embedding 反向执行器

改动只在 op-plugin 的 `op_api::embedding_dense_backward` 操作接口插入受控分支。实验使用一份二进制和相同的 512B 地址族视图。A 的 `NPU_DYNAMIC_REPEAT_EMBEDDING=fresh` 每笔重新建描述符并调用 `GetWorkspaceSize`。B 的值为 `repeat`，先设置一次 repeatable，后续同族只改两个输入和一个输出的 storage 地址，立即在原物理 stream 上调二阶段。两腿都在同一个 PTA consumer 回调内准备和下发，使用同一 workspace 分配策略。默认值或 0 走未改 stock 分支。

当前实现要求 Mode 1、无图捕获、单卡首个参与的流，`[B,L,1024,4]` BF16 训练对应的 int64 只读索引切片，B=1–4、L=12–32；表尺寸为 64 个行和 64 维，padding_idx=-1、scale_grad_by_freq=False。梯度、索引和输出必须是受标准 NPU allocator 所有的合格基格式，参数必须满足实际 stride、连续梯度/输出、完整 storage 长度、基址对齐和设备一致的检查。算子参数、地址族或 ABI 不适用时留在 stock；已成功准备但 runtime 不支持 repeatable 时用一次普通二阶段消费当前 executor，并将后续请求退回。stage2 失败直接上报，防止重放部分已提交的核。

consumer 中对一个 epoch 的 `[B,L,当前 Cube 核配额,Vector 核配额]` 维护至多 16 个 residual slot。不同流/设备不会复用其中的 executor。现有设备可见输入、输出由 queue 回调持有至 Host 提交，跨分配流通过 PTA allocator 的 `recordStream` 保活。队列工作在原 stream 有序执行。A、B 的 16 个槽共用首个参与流的一份私有工作区，同流有序提交。首次按 CANN 返回的实际字节数分配，后续需求增大时同步该流并更新缓冲。在形状或核配额变化时，同步旧 epoch，销毁 executor 和描述符，保留缓冲直到 finalize。释放回调注册为 NPU finalize 的 PriorityFirst，在销毁流之前排空和释放。运行计数可见 drain 次数、活跃槽位和累计峰值 workspace。通用 PTA 临时 workspace 可能在其他请求扩容时被释放，保留这种临时 allocator 的旧指针不可作为驻留工作区。

重新创建描述符时，候选同步修正 storage base、residual offset 和 bounded extent。raw-family 使用不同的 stock 索引描述符，实机原始探针发现 Slice→AsStrided，故需要用 bounded A、B 单独判断是否等核等量下发。fresh→repeat 衡量缓存执行器这一机制；stock PTA 自身有 executor cache，和 stock 之间的端到端差值要独立比较。[独立 ABI 探针与真实核核对方法](../README.md)

```bash
python <integration>/build_repeat.py --builder <dynamic-infra>/build_native.py --output <new-package-root>
export PYTHONPATH=<new-package-root>/package:${PYTHONPATH}
NPU_DYNAMIC_REPEAT_EMBEDDING=fresh python native_gate.py <fresh-gate.json>
NPU_DYNAMIC_REPEAT_EMBEDDING=repeat python native_gate.py <repeat-gate.json>
```

正常微测每腿各 45 个合格请求、1 次 rank-2 回退。A 预计 45 次 prepare，B 预计 15 次 prepare、30 次 reuse。正式运行需另查逐步激活、真实设备提交和输入 shape，再核验八步训练语义及双序无 profiler / Level0。

## 原动态 HSTU 的同轮配对

在原训练容器内构建时，`build_repeat.py --builder /artifacts/dynamic-infra-20261127/build_native.py --output "$BUILD"` 只重编译 embedding 入口的独立 TU。常规 installer 路径也可通过 `build_native.py --installer <本目录>/apply_repeat.py --output "$BUILD"` 复现。用同一构建的 `fresh` 和 `repeat` 走 `native_gate.py`，从 45 次请求核实合格/回退、工作区字节及地址重绑的实际值。

宿主机从已有 `dynamic-infra-20261127` 实验根目录执行以下命令。`PACKAGE` 是新构建的容器内绝对包路径，`integration` 中的 `stage.py` 和 `repeat_counts.py` 放在 `dynamic-free70/repeat_executor/integration/`。保留 01 分块 overflow、Mode 1、原始 60 个动态 eager shape。两腿均采用原 ZeRO 分区和 norm。由原 runner 串行持卡并进行卡况、输入、训练正确性核验。

```sh
set -e
study=/data/recsys-gr-ds019-host88/artifacts/dynamic-infra-20261127
cd "$study"
t=dynamic-free70/runner_tools
j=dynamic-free70/repeat_executor/integration
w="$study/20261127T033119Z-baseline-l1-341x7gd3/workspace"
r="$t/recipes/repeat-embedding-stream-workspace-v2.json"
python3 "$j/make_recipe.py" --package "$PACKAGE" --stage repeat_executor/integration --output "$r"
python3 "$t/run.py" "$r" gate --workspace "$w"
python3 "$t/run.py" "$r" l1/ab no-profiler/ab no-profiler/ba l0/ab l0/ba
python3 "$t/report.py" "$r"
python3 "$t/collect.py" "$r"
```

每步 A 预期 1,024 次 fresh；B 预期 16 次 fresh 和 1,008 次 reuse。两腿各有三张模型固有 embedding 的守卫回退。上述数字作为激活门限，实际值写入 `rank000/repeat_embedding_counts.json`。纳入收益前核验同算子、同流、核名和物理提交次数。fresh→repeat 检验复用机制；如果它通过，还应使用独立名称再生成一份 `make_recipe.py --baseline 0`，在真正 stock 路由与 repeat 间再次完成相同的语义及性能配对。stock→repeat 可能改变 ACLNN 内部 Slice/AsStrided，按对齐到真实核序列的结果说明两腿差异。
