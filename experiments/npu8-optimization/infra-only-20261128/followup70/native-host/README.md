# TorchNPU eager view 使用内联尺寸和步长

双向 embedding 描述符规范化和 `ACLNN_CACHE_LIMIT=65536` 同时开启时，原始 HSTU 的 [1,19] 和 [2,14] 两步各出现 9,545 次 `aten::view`。它们独占 producer 线程的 CPU 时间为 24.61 和 24.30 ms。按后继设备工作关联唯一 queue ID，再按该请求的真实 producer 线程，在「设备空档开始至后继进入 ATen」内只累计最内层 CPU 标记，得到 9.69 和 7.63 ms 的 view 交集。两步 view 的 p50/p95/p99/最大分别为 2.45/4.18/6.75/47.22 和 2.50/4.25/5.82/36.46 µs。这一边界包括 dispatcher 与 RecordFunction 的部分成本。[逐步结果](exposure.json) · [时间线复算](exposure.py)

原 01 默认缓存基线对同两组形状比较首次和再次执行。首见 #4/#5 的 view p95 为 9.57/9.50 µs，再见 #12/#13 为 6.00/11.06 µs，每步均 9,545 次。相同两组形状的变化方向不同。该对照可检查形状首见效应，运行的前序 RNG 和参数状态随步序演进。[首次](first-stock-exposure.json) · [再见](repeat-stock-exposure.json)

PTA `38e483ee79ec8a3286517808740a811c0f0ae66c` 的 `NPUNativeFunctions::view` 调用 `at::infer_size` 和返回 `std::vector` 的 `computeStride`，再复制其结果及输入 Tensor 句柄。候选对请求秩在 `c10::kDimVectorStaticSize` 以内的非量化输入使用 `infer_size_dv` 和 `computeStride(...,const DimVector&)`。两条 API 使用与原操作相同的 ATen 尺寸校验、步长推断及原生 alias 构造，后者按原张量设置 storage offset、strides 和 names。大秩和量化输入保留原路径。改变的工作在 producer 上计算视图元数据，预计设备提交不变。若视图独占耗时或配对端到端没有稳定改善，则分别报告。 [原生补丁](view.patch) · [固定源码构建回执](v1/build-receipt.json) · [与双向 embedding 的单对象重链](combined/receipt.json)

使用单独无卡 runc 容器构建，加载库 SHA‑256 为 `5c01592cfc911117e6644cfd823d49939e9a99ae25f2055e843b8fcc14342bcf`。A/B 唯一机制开关为 `TORCH_NPU_VIEW_INLINE_DIMS=0/1`。八步诊断两腿共用 `TORCH_NPU_VIEW_INLINE_DIMS_AUDIT=1`，计时与 L0/L1 均置为 0。每腿共同开启 01 overflow、双向 embedding、CANN 缓存 65536、Task Queue Mode 1。模型、训练和 runner 按原文件清单核查哈希。[串行运行程序](series.py)

独立语义微测对 60 个合成形状逐项检查 offset 非零的视图、storage 别名、stride、原地写的版本传播、数据及反向梯度。补测标量、零元、转置、单例非标准步长、大秩回退和五种错误。独立调用原 01 扫描确认有限梯度更新、Inf/NaN 梯度跳过更新。八步真实训练 A/B 各处理 88,979 个符合条件的 view，A 进入新路径 0 次、B 进入 88,979 次。两腿各完成 8 次有限更新，梯度和范数、权重和 AdamW 状态相同，loss 最大绝对差 5.72×10⁻⁶，通过原 runner 数值门槛。[微测代码](micro.py) · [A](micro-0/micro.json) · [B](micro-1/micro.json) · [真实八步回执](audit.json)

独立无卡容器直接调用原 wheel 的 libtorch 尺寸与 stride 推断 API。遍历 B=1–4、L=12–32 的 84 个组合，每次将连续 flat 视图推断为 [B,−1,1]，仅在推断和读取结果期间拦截 C++ `operator new`。stock 路径 252 次，候选 0 次。该微测量化了这一变换的临时堆分配。真实训练内还有其他 view shape，独立结果不能乘以全步调用数推算耗时。[测试源码](metadata_allocations.cpp) · [输出](allocations.json)

正式对照的 A 使用同一原生实验库中的原 view 实现。两腿均执行缓存开关检查和低秩非量化资格判断，B 才进入内联尺寸路径。所得差值以相同实验库的关状态为分母。

两步各 9,545 次 view 的 producer 独占 CPU 分别减少 3.970、3.256 ms，与后继入队前设备空档的交集减少 1.615、0.916 ms。三个物理 stream 的工作类型和名称按序完全吻合（17,220/17,220、17,221/17,221），ATen 逐线程顺序吻合，第 4 步 B 多一条 Host `Enqueue@destroy_event`。单次 ACL kernel launch p95 反而从 7.85/7.84 增至 9.96/9.87 µs。裸跑两序分别慢 0.132% 和 2.218%，Free 占比相对变化方向相反。此改动减少了视图元数据工作，实测端到端和 Free 未受益，归入「尝试无优化路径」。[八列、吞吐与 Level1](实机结果.md) · [设备工作逐项及空档关联](linked-level1.tar.gz)
