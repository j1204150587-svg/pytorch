# v2.10 指定分支整包构建

从 GitHub 核对 `codex/hstu-dynamic-eager-infra-v2.10` 的 head 为 `e3cda6399cb194c813530222478ea018f5aca5c9`。其相对 `38e483ee79ec8a3286517808740a811c0f0ae66c` 的三个提交依次为 `2cd525d5563cea44fb17596830c3a2367a99dff9`、`645b89a69c34509e28775b82a5bc478fb33ca556`、`e3cda6399cb194c813530222478ea018f5aca5c9`。在 host88 的无 NPU、无网络 runc 容器中，以训练镜像 `sha256:c342a038f6476c794fb364d96479286ebd456021e86dba0d3b715ce4cb983e33` 编译并校验；本项没有运行训练或 NPU 算子。[远端分支](https://github.com/j1204150587-svg/pytorch/tree/codex/hstu-dynamic-eager-infra-v2.10) · [引用校验](github-reference.txt) · [提交链](commits.txt)

### 产物与加载路径

| 产物 | 容器内绝对路径 | SHA-256 |
|---|---|---|
| 训练 PYTHONPATH | `/artifacts/infra-only-20261128/branch-stability-v210/build/overlay` | 其中 2,378 个文件逐项对照 wheel |
| native ELF | `/artifacts/infra-only-20261128/branch-stability-v210/build/overlay/torch_npu/lib/libtorch_npu.so` | `175bd91575354569a63d8599a501b0a19a2a224ada114e44a763ba38fa817d75` |
| 完整 wheel | `/artifacts/infra-only-20261128/branch-stability-v210/build/wheel/torch_npu-2.10.0.post6-cp312-cp312-manylinux_2_28_aarch64.whl` | `5157e7bf7c01fdf00e9c520fef2c27c52cd8beeea5bac71270bd18db24b2e6d7` |
| 就绪收据 | `/artifacts/infra-only-20261128/branch-stability-v210/build/ready.json` | 含版本、导入路径、导出符号、映射库与检查结果 |

宿主机对应绝对路径为：

```text
/data/recsys-gr-ds019-host88/artifacts/infra-only-20261128/branch-stability-v210/build/overlay
/data/recsys-gr-ds019-host88/artifacts/infra-only-20261128/branch-stability-v210/build/overlay/torch_npu/lib/libtorch_npu.so
/data/recsys-gr-ds019-host88/artifacts/infra-only-20261128/branch-stability-v210/build/wheel/torch_npu-2.10.0.post6-cp312-cp312-manylinux_2_28_aarch64.whl
/data/recsys-gr-ds019-host88/artifacts/infra-only-20261128/branch-stability-v210/build/ready.json
```

### 编译与依赖复核

将已验证的三个提交以 Git bundle 应用到专用缓存副本。副本中的两份旧修改为 `NPUQueue.cpp` 和 `OpCommand.cpp`，两者都从正确源码重新编译。op-plugin 指向分支 gitlink `eaf6b55812a1998a4a40e485e3b2861a40e9c2f6`。在单独容器内将副本映射到原 CMake 路径；实际执行 `cmake -S "$core" -B "$core/build"` 及 `cmake --build "$core/build" --target torch_npu -j8 -v`。分支 `embedding.cmake` 从原 op-plugin TU 或已有 codegen 输出复制、应用补丁，再替换 CMake 目标中的输入。配置文件、三个补丁动作、编译 command 和真实 link argv 均已保留。[宿主启动脚本](stage-build.sh) · [CMake 命令](configure-build.sh) · [Ninja 原始日志](compile-link.log)

实际链接 1,105 个对象。其中 5 个本轮重编，分别是 `op_api_common.cpp`、`StructKernelNpuOpApi.cpp`、`EmbeddingKernelNpuOpApi.cpp`、`NPUQueue.cpp`、`OpCommand.cpp`；其余 1,100 个逐项核对与种子构建的 SHA-256 相同。未取用既有独立候选组合 ELF。Ninja 复用了基线代码生成文件和同版缓存参数，其中编译期版本字符串保留 `2.10.0.post6+git38e483e`；wheel 的真实 `torch_npu.version.git_version` 为本轮分支 head。三个未初始化的 ACL 源码子模块以及已生成的兼容构建材料沿用种子环境。[编译来源](build-provenance.json) · [1,100 个对象指纹](reused-objects.json) · [子模块状态](submodules.txt)

### Wheel、overlay 和加载门

打包复用了镜像中与原装 wheel RECORD 对应且逐项通过 SHA-256 和大小校验的原始材料。wheel 仅有两个 payload 文件变化，即本轮新 ELF 以及标明本分支 SHA 的 `torch_npu/version.py`。打包后重读整个 wheel 及 RECORD，离线 pip 安装到独立 `wheel-install`，再从真实安装物生成只含 `torch_npu/` 的训练根；无 pyc。对全部 2,440 个 wheel payload、overlay 的全部 2,378 个 `torch_npu` 文件逐项校验 SHA；另从真实训练容器的只读包目录核对所有原文件，与 overlay 的变化恰为这两项。版本文件仅替换一处 git SHA，原/新文件 SHA 分别为 `4bd3123f98c6bb957e1e36db50bf5d506992e5c890b6bd4654d0c1460b1788e4`、`2e22e41c20a56155a93fbc4a697b4b1710230e7e901484211a5423e85bf30da8`。[打包脚本](package_branch.py) · [全部 payload 对照](package-payload.json)

独立无卡进程实际 import torch_npu，校验版本、`_C`、映射中唯一的 libtorch_npu.so、双向 embedding 计数符号和队列审计标志。上述检查全部通过后原子发布 ready.json。打包容器首次因 OCI 不支持 /proc 路径 bind 而未启动，重试使用经 dev/inode 验证等价的训练容器 merged 目录，打包及校验 exit=0。[加载校验](verify_overlay.py) · [就绪收据](ready.json) · [成功日志](package2.log) · [完整构建证据包](build-evidence.tgz)
