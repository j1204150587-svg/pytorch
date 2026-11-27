# HSTU 动态 eager 的原生提交实验

实测运行环境为 Ascend 910、PyTorch 2.10.0、TorchNPU 2.10.0.post6、DeepSpeed 0.19.0 和 CANN 9.1.0。训练工作区由相邻的 `dynamic-shapes-20261127` 目录提供。各原生包、环境开关和计时窗口的定义在本目录。所有正式配对只切换对应的一个开关。

| 配方 | 运行时改动 | 候选实现 |
|---|---|---|
| cache-limit.json | CANN executor 缓存上限 | stock wheel 的 `ACLNN_CACHE_LIMIT=65536` |
| index-rebase.json | Mode 1 下为只读 embedding 反向索引规范化描述符 | `rebase/apply_rebase.py` |
| unflatten.json | 为 ZeRO 回写直接构造受守卫的 BF16 视图 | `provenance/unflatten/unflatten.patch` |
| cross-copy.json | 保持同流 #42 开启，切换对其他分配流来源的 SDMA 及存活保护 | `cross_copy/from-stock.patch` |
| forward-rebase.json | 规范化 embedding 前向的只读 BF16 权重及索引 | `rebase_forward/apply_forward.py` |

## 复现

宿主使用训练容器 `recsys-gr-ds019-npu8` 和现有 `/artifacts` 挂载。将本目录部署在宿主 `/data/recsys-gr-ds019-host88/artifacts/dynamic-infra-20261127`。对于新的实验目录，把相邻动态训练目录复制为 `runner/`，运行 `python3 adapt_runner.py runner/run_npu8.py`；多任务排队时再运行 `python3 wait_for_slot.py runner/run_suite.py`。已有目录可沿用该目录经过适配的 runner。复现前由 runner 核对逻辑 8、容器状态，独占宿主共享锁。

```bash
cd /data/recsys-gr-ds019-host88/artifacts/dynamic-infra-20261127
(cd runner && python3 -c "import run_suite as s; raise SystemExit(s.run_pair('PARTITION','l1','baseline')['exit'])")
# 填入上一步返回的宿主运行目录。
export DYNAMIC_INFRA_WORKSPACE="<baseline-run>/workspace"
python3 run_candidate.py cache-limit no-profiler/ab no-profiler/ba l0/ab l0/ba l1/ab
python3 late_profile.py prepare --runner runner --out runner-late
python3 late_profile.py run --runner runner-late --recipe cache-limit.json --order ab --result cache-late-ab.json
```

`build_native.py`、`rebase_forward/build_forward.py` 和 `cross_copy/build_copy.py` 在编译容器内复用固定的旧 Ninja build，替换受影响的目标文件后重链隔离的 Python 包。两腿用同一份包，运行时加载的 SHA 与构建收据交叉核对。各 JSON 配方记录容器内 PYTHONPATH。D03 复用在机已编译且核对过的包，归档含原补丁、重链脚本和加载库指纹。正式计时和 Level0/1 均关闭缓存审计；诊断专用 `LD_AUDIT` 程序独立运行。训练输入取标准 60 个不同的实际尺寸。相同形状再访的 8+8 仅用于解释缓存机制。

`semantic/run_semantic_pair.py` 进行 8 步真实动态训练的梯度、权重与溢出跳步核验。`rebase/micro_gate.py`、`rebase_forward/micro_gate.py` 和 `cross_copy/run_gate.py` 还覆盖各自的输入守卫和回退。`reporting/recompute_pairs.py` 从无 profiler 的同步窗口与 Level0 原始 CSV 计算八列。`analysis/compare_dynamic_legs.py` 复核 Level1 的队列与逐流设备提交。`reporting/collect_receipts.py` 生成包含原始 trace、命令与 SHA-256 的证据包。

各轮输出使用新目录。再次运行时使用新的 gate 和 result 路径；复算可直接读取已完成的运行。

