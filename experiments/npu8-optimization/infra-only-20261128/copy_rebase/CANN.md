# 为什么相同 key 仍重新准备 InplaceCopy

本机容器使用 CANN 9.1.0，`/usr/local/Ascend/cann/lib64/libnnopbase.so` 的 SHA-256 为 `308dda461704a6e5f70de6e0eac4495e9829b9cec3328563b75a5b382c6e438b`。以下结论限于同 dtype 单 SDMA 分支。

1. PTA `hit_cache` 先检查 `CanUsePTACache(aclnn_api)`，再执行 `add_param_to_buf`。8 步训练中修订版记录 32,768 个 copy 参数 key，证明该 API 通过粗粒度准入。随后 32,768 次参数转换都新建了 descriptor。相同进程中，原有双向 embedding 出现真正的 cache hit。
2. 运行库 `op::CopyNpuToNpu` 在地址 `0x2276b4` 调用 `aclOpExecutor::AddToKernelLauncherListCopyTask`。后者在 `0x1cbd0c` 设置布尔实参为 0，在 `0x1cbd30` 调用 `OpExecutorImpl::AbandonCache(false)`。这个布尔参数控制函数的另一项状态写入。函数对已有的非空 OpExecCache 总是在 `0x1cb5f8` 调用 `MarkOpCacheInvalid()`。这条路径解释了为什么规范化地址和偏移后，单 SDMA 路径仍反复进入 GetWorkspaceSize。[CopyNpuToNpu 的调用关系](cann-cache-symbols.txt) · [登记 copy 任务](cann-add-copy-task.txt) · [放弃普通缓存](cann-abandon.txt) · [缓存完成与添加符号](cann-finalize.txt)
3. 独立的 repeatable executor 接口调用 `CopyNpuToNpuKernelLauncher::CheckRepeatable`，该函数在 `0x2281f0` 返回 0。这个判断只说明该 launcher 不支持 repeatable，普通缓存的证据在上一项。[运行库指纹和函数](cann-repeatability.txt)
