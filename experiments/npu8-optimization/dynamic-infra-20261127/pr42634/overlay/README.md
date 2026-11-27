# GitCode !42634 的隔离构建与激活核查

PR 目标分支为 `Ascend/pytorch:v2.10.0`，源分支为 `linjiyuan/pytorch_ascend_origin:codex/fast-launch-patch-v2.10.0`。功能提交是 `b1c30ebd929bd381fdb14278657bb7b3da692701`，文档提交和 MR head 同为 `2701f7b3e8d89625752edbde10e42c2b2172272c`，功能提交的父为 `9412cbac3f7fbf45d24ac2b8fd1a823935bf2e8c`。两提交合计 18 个文件、3,119 行新增。从这两个端点生成 `git diff --binary --full-index` 得 119,828 字节，SHA-256 为 `39a964222957514079f93f0441b3512d29d8d2dc87c98f4a5432de0e5334c95e`。[PR](https://gitcode.com/Ascend/pytorch/pull/42634) · [提交 API](https://api.gitcode.com/api/v5/repos/Ascend/pytorch/pulls/42634/commits) · [固定版本 bindings.cpp](https://raw.gitcode.com/Ascend/pytorch/raw/2701f7b3e8d89625752edbde10e42c2b2172272c/torch_npu/_inductor/experimental/python_wrapper_fast_launch/csrc/bindings.cpp)

已知训练容器为 TorchNPU 2.10.0.post6、PyTorch 2.10.0、CANN 9.1。之前保留的 PTA 构建回执记录 checkout `38e483ee79ec8a3286517808740a811c0f0ae66c`。这个版本的 codegen 类型和路径为 `NPUWrapperCodeGen`、`torch_npu._inductor.npu_triton_heuristics`。PR 使用 `NPUPythonWrapperCodeGen`、`torch_npu._inductor.runtime.triton_heuristics`。将原始补丁对旧树执行 `git apply --check` 时有四处入口不匹配。独立包使用同一 commit 的完整 Inductor 与所需 Dynamo Python 源，避免混搭两个版本的 codegen。

原实现的 `bindings.cpp` 调用 `OpCommand.Name().SetCustomHandler().Run()`。非 SIMT 通过 `rtKernelLaunch` 下发，SIMT 使用弱引用 `rtKernelLaunchWithFlagV2`，按需查询 FFTS 地址。关键头文件 `OpCommand.h` 与 `hacl_rt.h` 在旧 PTA checkout 和 PR head 的 git blob 完全一致。构建脚本以安装的 torch ABI、已有 compile_commands 的 include、stock libtorch_npu 和实际 CANN libruntime 编译原函数，并将其三个入口注册到触发路径使用的 `torch_npu._C`。创建隔离包时拷贝或硬链接原文件，修改时替换目录项。两腿都应从同一 overlay 启动，保留每次载入路径和符号回执。

```sh
p=/artifacts/dynamic-infra-20261127/dynamic-free70
src="$p/pr42634_overlay"
out="$p/pr42634-package-v1"
python3 "$src/smoke.py" inventory "$p/stock-inventory.json"
python3 "$src/build.py" --output "$out"
python3 "$src/complete_frontend.py" "$out"
export PYTHONPATH="$out/package:${PYTHONPATH:-}"
export TORCHINDUCTOR_NPU_FAST_LAUNCH=1
python3 "$src/smoke.py" abi "$p/pr42634-abi.json"
python3 "$src/smoke.py" frontend "$p/pr42634-frontend.json"
python3 "$src/smoke.py" eager "$p/pr42634-eager-control.json"
python3 "$src/smoke.py" static "$p/pr42634-compiled-control.json"
python3 "$src/smoke.py" dynamic "$p/pr42634-dynamic-control.json"
```

`abi` 只创建 Host metadata，不提交设备工作。`eager` 检查普通 BF16 eager 运算是否激活。两个 compiled control 独立使用 Inductor，结果标记为编译子场景。每步同时核对位级输出和原版两个 C 入口的调用及成功计数。单次 `frontend_patched` 或只有 plan 创建不能证明真实提交生效。原实现中 profiler 会走原 dispatch，grouped autotuner 会绑定原入口，只有受支持的 Triton Python Wrapper 生成的 callsite 能走计划。比较拟归因的性能时需专门检查无 profiler 的成功提交计数。[原实现使用说明](https://gitcode.com/Ascend/pytorch/blob/2701f7b3e8d89625752edbde10e42c2b2172272c/torch_npu/_inductor/docs/npu_inductor_fast_launch_zh.md)
