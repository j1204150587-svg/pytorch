#include "op_plugin/utils/mm_rebase.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace projection_rebase {
namespace {
thread_local const View* active = nullptr;
std::atomic<uint64_t> accepted{0}, rejected{0}, keys{0}, descriptors{0};

bool experiment_enabled()
{
    static const bool enabled = [] {
        const char* flag = std::getenv("NPU_DYNAMIC_MM_REBASE_128");
        return flag && std::strcmp(flag, "1") == 0;
    }();
    return enabled;
}

bool eligible(const at::Tensor& input, const at::Tensor& weight)
{
    return input.scalar_type() == at::kBFloat16 && weight.scalar_type() == at::kBFloat16 &&
        input.device() == weight.device() && input.layout() == at::kStrided && weight.layout() == at::kStrided &&
        input.dim() == 2 && weight.dim() == 2 && input.size(0) >= 1 && input.size(0) <= 128 &&
        input.size(1) == weight.size(0) &&
        ((weight.size(0) == 256 && weight.size(1) == 64) || (weight.size(0) == 64 && weight.size(1) == 192)) &&
        input.is_contiguous() && weight.is_contiguous() && weight.storage_offset() > 0 &&
        !weight.is_neg() && !weight.is_conj() && at_npu::native::FormatHelper::IsOpInputBaseFormat(weight);
}
}

Scope::Scope(const at::Tensor& input, const at::Tensor& weight) : previous_(active)
{
    active = nullptr;
    if (!experiment_enabled()) {
        return;
    }
    if (eligible(input, weight) && c10_npu::option::OptionsManager::GetTaskQueueEnable() == 1 &&
        c10_npu::currentStreamCaptureStatus() == c10_npu::CaptureStatus::None) {
        const auto base = reinterpret_cast<uintptr_t>(weight.storage().data());
        const auto address = reinterpret_cast<uintptr_t>(weight.data_ptr());
        const auto capacity = weight.storage().nbytes();
        const auto bytes = static_cast<size_t>(weight.numel()) * weight.itemsize();
        if (base != 0 && base % 512 == 0 && address >= base && address % 128 == 0 &&
            address - base <= capacity && bytes <= capacity - (address - base)) {
            view_ = {weight.unsafeGetTensorImpl(), reinterpret_cast<void*>(address), weight.numel()};
            active = &view_;
            accepted.fetch_add(1, std::memory_order_relaxed);
            return;
        }
    }
    rejected.fetch_add(1, std::memory_order_relaxed);
}

Scope::~Scope() { active = previous_; }
const View* lookup(const at::Tensor& tensor)
{
    return active && active->owner == tensor.unsafeGetTensorImpl() ? active : nullptr;
}

extern "C" __attribute__((visibility("default"))) uint64_t npu_mm_rebase_counter(unsigned index)
{
    switch (index) {
        case 0: return accepted.load(std::memory_order_relaxed);
        case 1: return rejected.load(std::memory_order_relaxed);
        case 2: return keys.load(std::memory_order_relaxed);
        case 3: return descriptors.load(std::memory_order_relaxed);
        default: return 0;
    }
}
}
