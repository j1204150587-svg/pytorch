#include "op_plugin/utils/dynamic_index_scope.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace dynamic_index_rebase {
namespace {
thread_local const View *active = nullptr;
std::atomic<uint64_t> eligible{0}, fallback{0}, key_uses{0}, descriptor_uses{0};

bool make_view(const at::Tensor &indices, View &result)
{
    if (indices.dim() != 3 || indices.size(2) != 4 || indices.size(0) <= 0 || indices.size(1) <= 0 ||
        indices.scalar_type() != at::kLong || indices.layout() != at::kStrided ||
        indices.stride(2) != 1 || indices.stride(1) < 4 || indices.stride(0) <= 0 ||
        indices.storage_offset() < 0 || indices.is_neg() || indices.is_conj() ||
        !at_npu::native::FormatHelper::IsOpInputBaseFormat(indices) ||
        c10_npu::option::OptionsManager::GetTaskQueueEnable() != 1 ||
        c10_npu::currentStreamCaptureStatus() != c10_npu::CaptureStatus::None) {
        return false;
    }
    // Advance the descriptor base only in multiples of 512 bytes. Keep the
    // original base alignment and the residual index offset that kernels see.
    constexpr int64_t item = sizeof(int64_t);
    constexpr int64_t quantum = 512 / item;
    const auto bytes = indices.storage().nbytes();
    if (bytes % item || bytes / item > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    const int64_t elements = static_cast<int64_t>(bytes / item);
    const int64_t original = indices.storage_offset();
    const int64_t shift = original - original % quantum;
    const int64_t residual = original - shift;
    if (original >= elements || indices.stride(0) / indices.size(1) < indices.stride(1)) {
        return false;
    }
    // Build residual + 1 + sum((size - 1) * stride) without overflowing or
    // crossing the actual storage. Round the view extent within this storage.
    int64_t length = residual + 1;
    const int64_t available = elements - shift;
    for (int64_t axis = 0; axis < indices.dim(); ++axis) {
        const int64_t size_minus_one = indices.size(axis) - 1;
        const int64_t stride = indices.stride(axis);
        if (size_minus_one > (available - length) / stride) {
            return false;
        }
        length += size_minus_one * stride;
    }
    const int64_t padding = (quantum - length % quantum) % quantum;
    if (padding > available - length) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(indices.storage().data());
    const uint64_t advance = static_cast<uint64_t>(shift) * item;
    if (!base || base % 512 || advance > std::numeric_limits<uintptr_t>::max() - base) {
        return false;
    }
    result.tensor = indices.unsafeGetTensorImpl();
    result.base = reinterpret_cast<void *>(base + advance);
    result.offset = residual;
    result.storage_elements = length + padding;
    return true;
}
}  // namespace

Scope::Scope(const at::Tensor &indices) : previous_(active)
{
    active = nullptr;
    static const bool enabled = [] {
        const char *value = std::getenv("NPU_DYNAMIC_REBASE_INDICES");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (enabled) {
        if (make_view(indices, current_)) {
            active = &current_;
            eligible.fetch_add(1, std::memory_order_relaxed);
        } else {
            fallback.fetch_add(1, std::memory_order_relaxed);
        }
    }
}

Scope::~Scope()
{
    active = previous_;
}

const View *for_tensor(const at::Tensor &tensor)
{
    return active && active->tensor == tensor.unsafeGetTensorImpl() ? active : nullptr;
}

extern "C" __attribute__((visibility("default"))) uint64_t npu_dynamic_index_rebase_counter(unsigned index)
{
    switch (index) {
        case 0: return eligible.load(std::memory_order_relaxed);
        case 1: return fallback.load(std::memory_order_relaxed);
        case 2: return key_uses.load(std::memory_order_relaxed);
        case 3: return descriptor_uses.load(std::memory_order_relaxed);
        default: return 0;
    }
}
}  // namespace dynamic_index_rebase
