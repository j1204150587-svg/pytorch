"""Generate the opt-in queued copy/replay patch against the measured TorchNPU commit."""

from difflib import unified_diff
import os
from pathlib import Path
import subprocess


BASE = "38e483ee79ec8a3286517808740a811c0f0ae66c"
CORE = "torch_npu/csrc/core/npu/NPUGraph"
GRAPH = "torch_npu/csrc/npu/Graph.cpp"
FLATTEN = "torch_npu/csrc/aten/ops/FlattenDenseTensorsKernelNpu.cpp"


def update(source, old, new):
    if source.count(old) != 1:
        raise RuntimeError(f"expected precisely one patch anchor: {old}")
    return source.replace(old, new)


def edit_header(source):
    source = update(source, "#include <ATen/Tensor.h>", "#include <optional>\n\n#include <ATen/Tensor.h>")
    source = update(source, "    void replay();", """    void replay();
    bool try_replay_with_copy(const at::Tensor& dst, const at::Tensor& src);""")
    return update(source, "    int capture_dev_;", """    int capture_dev_;

    // The private copy/replay path uses at most one replay stream.
    std::optional<NPUStream> queued_replay_stream_;
    void finish_queued_replays();""")


def edit_graph(source):
    source = update(source, '#include "torch_npu/csrc/core/npu/NPUFunctions.h"', '''#include "torch_npu/csrc/core/npu/NPUFunctions.h"
#include "torch_npu/csrc/core/NPUBridge.h"
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include "torch_npu/csrc/framework/OpCommand.h"''')
    source = update(source, "void NPUGraph::enable_debug_mode()", """bool NPUGraph::try_replay_with_copy(const at::Tensor& dst, const at::Tensor& src)
{
    TORCH_CHECK(has_graph_exec_, "Caught copy/replay before successful graph capture.");
    if (!dst.defined() || !src.defined() || dst.device() != capture_stream_.device() ||
        src.device() != dst.device() || dst.sizes() != src.sizes() || dst.scalar_type() != src.scalar_type() ||
        !dst.is_contiguous() || !src.is_contiguous() || dst.storage_offset() != 0 || src.storage_offset() != 0 ||
        dst.is_conj() || dst.is_neg() || src.is_conj() || src.is_neg() || src.numel() == 0) {
        return false;
    }
    const auto* allocator = NPUCachingAllocator::get();
    const auto dst_storage = dst.storage();
    const auto src_storage = src.storage();
    if (dst_storage.allocator() != allocator || src_storage.allocator() != allocator ||
        dst_storage.unsafeGetStorageImpl() == src_storage.unsafeGetStorageImpl() ||
        torch_npu::NPUBridge::GetNpuStorageImpl(dst)->npu_desc_.npu_format_ != ACL_FORMAT_ND ||
        torch_npu::NPUBridge::GetNpuStorageImpl(src)->npu_desc_.npu_format_ != ACL_FORMAT_ND) {
        return false;
    }
    if (option::OptionsManager::GetTaskQueueEnable() != 1 || option::OptionsManager::GetPerStreamQueue()) {
        return false;
    }
    c10::OptionalDeviceGuard guard{capture_stream_.device()};
    auto stream = c10_npu::getCurrentNPUStream();
    if (c10_npu::detail::isExternalStream(stream) || stream.isSyncLaunchStream() ||
        (queued_replay_stream_ && *queued_replay_stream_ != stream) ||
        currentStreamCaptureStatusMayInitCtx() != CaptureStatus::None) {
        return false;
    }
    for (const auto& captured_generator : captured_generator_states_) {
        if (captured_generator.second != 0) {
            return false;
        }
    }

    const size_t bytes = src.nbytes();
    const void* source = src.const_data_ptr();
    void* target = dst.mutable_data_ptr();
    const auto model = model_ri_;
    const auto raw_stream = stream.stream(false);
    NPUCachingAllocator::recordStream(src_storage.data_ptr(), stream);
    NPUCachingAllocator::recordStream(dst_storage.data_ptr(), stream);
    queued_replay_stream_ = stream;
    at_npu::native::OpCommand::RunOpApiV3(
        "NPUGraphCopyReplay", [source, target, bytes, model, raw_stream]() -> int {
            auto ret = aclrtMemcpyAsync(target, bytes, source, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, raw_stream);
            if (ret != ACL_ERROR_NONE) {
                return ret;
            }
            return c10_npu::acl::AclmdlRIExecuteAsync(model, raw_stream);
        }, false, &stream);
    return true;
}

void NPUGraph::finish_queued_replays()
{
    if (queued_replay_stream_) {
        // Synchronize both the PTA queue and its device stream before releasing the graph or its pool.
        queued_replay_stream_->synchronize();
        queued_replay_stream_.reset();
    }
}

void NPUGraph::enable_debug_mode()""")
    source = update(source, """void NPUGraph::super_kernel_optimize(const aclskOptions *options)
{""", """void NPUGraph::super_kernel_optimize(const aclskOptions *options)
{
    finish_queued_replays();""")
    source = update(source, """void NPUGraph::reset()
{""", """void NPUGraph::reset()
{
    finish_queued_replays();""")
    return update(source, "    reset();\n}\n", """    try {
        finish_queued_replays();
    } catch (const std::exception& error) {
        TORCH_WARN("Unable to safely release a queued NPU graph: ", error.what());
        return;
    }
    reset();
}
""")


def edit_binding(source):
    return update(source, '        .def(\n            "reset",', """        .def(
            "_try_replay_with_copy",
            torch::wrap_pybind_function_no_gil(&c10_npu::NPUGraph::try_replay_with_copy))
        .def(
            "reset",""")


def edit_flatten(source):
    source = update(source, "namespace at_npu {", """#include <cstdlib>
#include <cstring>
#include <ATen/record_function.h>

namespace at_npu {""")
    return update(source, "    return at::cat(c10::fmap(tensors, flatten));", """    static const bool enable_storage_copy = [] {
        const char* value = std::getenv("NPU_EXPERIMENT_FLAT_ND");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (enable_storage_copy && tensors.size() > 1) {
        const auto& first = tensors[0];
        const auto storage = first.storage();
        const auto* storage_impl = storage.unsafeGetStorageImpl();
        const auto& desc = torch_npu::NPUBridge::GetNpuStorageImpl(first)->npu_desc_;
        if (desc.npu_format_ == ACL_FORMAT_ND && desc.origin_format_ == ACL_FORMAT_ND) {
            const auto start = first.storage_offset();
            const auto capacity = storage.nbytes() / first.element_size();
            auto end = start;
            bool adjacent = start >= 0 && static_cast<size_t>(start) <= capacity;
            for (const auto& tensor : tensors) {
                if (!adjacent || !tensor.defined() || !tensor.has_storage() || tensor.device() != first.device() ||
                    tensor.scalar_type() != first.scalar_type() || tensor.requires_grad() || tensor._fw_grad(0).defined() ||
                    tensor.is_conj() || tensor.is_neg() || tensor.has_names() || tensor.is_quantized() ||
                    !tensor.is_contiguous() || tensor.storage().unsafeGetStorageImpl() != storage_impl ||
                    tensor.storage_offset() != end || tensor.numel() == 0 ||
                    static_cast<size_t>(tensor.numel()) > capacity - static_cast<size_t>(end)) {
                    adjacent = false;
                    break;
                }
                end += tensor.numel();
            }
            if (adjacent) {
                RECORD_FUNCTION("NPUDenseFlattenStorageCopy", std::vector<c10::IValue>());
                // Match cat's fresh storage, while avoiding one dispatch per chunk.
                return first.as_strided({end - start}, {1}, start).clone(at::MemoryFormat::Contiguous);
            }
        }
    }
    return at::cat(c10::fmap(tensors, flatten));""")


def main():
    paths = {f"{CORE}.h": edit_header, f"{CORE}.cpp": edit_graph, GRAPH: edit_binding, FLATTEN: edit_flatten}
    parts = []
    for path, edit in paths.items():
        initial = subprocess.check_output(
            ["git", "show", f"{BASE}:{path}"], env=os.environ | {"GIT_NO_LAZY_FETCH": "1"}
        ).decode()
        modified = edit(initial)
        parts.extend(unified_diff(initial.splitlines(True), modified.splitlines(True), f"a/{path}", f"b/{path}"))
    destination = Path(__file__).resolve().with_name("queued-copy-replay.patch")
    destination.write_text("".join(parts), encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
