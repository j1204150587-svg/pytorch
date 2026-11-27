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

bool eligible_indices(const at::Tensor &t)
{
    return t.layout() == at::kStrided && t.scalar_type() == at::kLong && t.dim() == 3 && t.size(2) == 4 &&
           t.size(0) > 0 && t.size(1) > 0 && t.stride(2) == 1 && t.stride(1) >= 4 &&
           t.stride(0) > 0 && t.stride(0) / t.size(1) >= t.stride(1);
}

bool eligible_weight(const at::Tensor &t)
{
    return t.layout() == at::kStrided && t.scalar_type() == at::kBFloat16 && t.dim() == 2 &&
           t.size(0) == 64 && t.size(1) == 64 && t.stride(0) == 64 && t.stride(1) == 1;
}

bool make_view(const at::Tensor &tensor, View &result)
{
    if (tensor.layout() != at::kStrided || tensor.storage_offset() < 0 ||
        tensor.is_neg() || tensor.is_conj() || !at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor)) {
        return false;
    }
    // Advance the descriptor base only in multiples of 512 bytes. Keep the
    // original base alignment and the residual index offset that kernels see.
    const int64_t item = tensor.itemsize();
    const int64_t quantum = 512 / item;
    const auto bytes = tensor.storage().nbytes();
    if (bytes % item || bytes / item > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
        return false;
    }
    const int64_t elements = static_cast<int64_t>(bytes / item);
    const int64_t original = tensor.storage_offset();
    const int64_t shift = original - original % quantum;
    const int64_t residual = original - shift;
    if (original >= elements) {
        return false;
    }
    // Build residual + 1 + sum((size - 1) * stride) without overflowing or
    // crossing the actual storage. Round the view extent within this storage.
    int64_t length = residual + 1;
    const int64_t available = elements - shift;
    for (int64_t axis = 0; axis < tensor.dim(); ++axis) {
        const int64_t size_minus_one = tensor.size(axis) - 1;
        const int64_t stride = tensor.stride(axis);
        if (size_minus_one > (available - length) / stride) {
            return false;
        }
        length += size_minus_one * stride;
    }
    const int64_t padding = (quantum - length % quantum) % quantum;
    if (padding > available - length) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(tensor.storage().data());
    const uint64_t advance = static_cast<uint64_t>(shift) * item;
    if (!base || base % 512 || advance > std::numeric_limits<uintptr_t>::max() - base) {
        return false;
    }
    result.tensor = tensor.unsafeGetTensorImpl();
    result.base = reinterpret_cast<void *>(base + advance);
    result.offset = residual;
    result.storage_elements = length + padding;
    return true;
}
}  // namespace

Scope::Scope(const at::Tensor &weight, const at::Tensor &indices) : previous_(active)
{
    active = nullptr;
    static const bool enabled = [] {
        const char *value = std::getenv("NPU_DYNAMIC_REBASE_FORWARD");
        return value && std::strcmp(value, "1") == 0;
    }();
    if (enabled) {
        if (eligible_indices(indices) && eligible_weight(weight) &&
            c10_npu::option::OptionsManager::GetTaskQueueEnable() == 1 &&
            c10_npu::currentStreamCaptureStatus() == c10_npu::CaptureStatus::None &&
            make_view(weight, views_[0]) && make_view(indices, views_[1])) {
            active = views_;
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
    if (active) {
        if (active[0].tensor == tensor.unsafeGetTensorImpl()) {
            return &active[0];
        }
        if (active[1].tensor == tensor.unsafeGetTensorImpl()) {
            return &active[1];
        }
    }
    return nullptr;
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
