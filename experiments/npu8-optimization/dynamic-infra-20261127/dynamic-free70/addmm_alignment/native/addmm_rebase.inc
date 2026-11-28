#include "op_plugin/utils/addmm_rebase.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include <atomic>
#include <cstdlib>
#include <cstring>

namespace projection_rebase {
namespace {
thread_local const View *active = nullptr;
std::atomic<uint64_t> accepted{0}, rejected{0}, keys{0}, descriptors{0};

bool experiment_enabled()
{
    static const bool enabled = [] {
        const char *flag = std::getenv("NPU_DYNAMIC_ADDMM_REBASE_128");
        return flag && std::strcmp(flag, "1") == 0;
    }();
    return enabled;
}

bool eligible_projection(const at::Tensor &bias, const at::Tensor &input, const at::Tensor &weight)
{
    if (bias.dim() != 1 || input.dim() != 2 || weight.dim() != 2 ||
        bias.device() != input.device() || bias.device() != weight.device() ||
        input.scalar_type() != at::kBFloat16 || bias.scalar_type() != at::kBFloat16 ||
        weight.scalar_type() != at::kBFloat16 || !input.is_contiguous() ||
        !bias.is_contiguous() || input.size(0) < 1 || input.size(0) > 128) {
        return false;
    }
    const int64_t k = input.size(1), n = bias.numel();
    return ((k == 64 && n == 256) || (k == 192 && n == 64)) &&
           weight.size(0) == k && weight.size(1) == n && weight.stride(0) == 1 && weight.stride(1) == k &&
           bias.storage().unsafeGetStorageImpl() == weight.storage().unsafeGetStorageImpl();
}

bool describe(const at::Tensor &tensor, View &view)
{
    if (tensor.layout() != at::kStrided || tensor.storage_offset() < 0 ||
        tensor.is_neg() || tensor.is_conj() || !at_npu::native::FormatHelper::IsOpInputBaseFormat(tensor)) {
        return false;
    }
    const auto base = reinterpret_cast<uintptr_t>(tensor.storage().data());
    const auto address = reinterpret_cast<uintptr_t>(tensor.data_ptr());
    const auto available = tensor.storage().nbytes();
    const auto extent = static_cast<size_t>(tensor.numel()) * tensor.itemsize();
    if (base == 0 || base % 512 || address < base || address % 128 ||
        address - base > available || extent > available - (address - base)) {
        return false;
    }
    view = {tensor.unsafeGetTensorImpl(), reinterpret_cast<void *>(address), tensor.numel()};
    return true;
}
}  // namespace

Scope::Scope(const at::Tensor &bias, const at::Tensor &input, const at::Tensor &weight)
    : previous_(active)
{
    active = nullptr;
    if (!experiment_enabled()) {
        return;
    }
    if (eligible_projection(bias, input, weight) &&
        c10_npu::option::OptionsManager::GetTaskQueueEnable() == 1 &&
        c10_npu::currentStreamCaptureStatus() == c10_npu::CaptureStatus::None &&
        describe(bias, views_[0]) && describe(weight, views_[1])) {
        active = views_;
        accepted.fetch_add(1, std::memory_order_relaxed);
    } else {
        rejected.fetch_add(1, std::memory_order_relaxed);
    }
}

Scope::~Scope()
{
    active = previous_;
}

const View *lookup(const at::Tensor &t)
{
    if (active) {
        for (int i = 0; i < 2; ++i) {
            if (active[i].owner == t.unsafeGetTensorImpl()) {
                return active + i;
            }
        }
    }
    return nullptr;
}

extern "C" __attribute__((visibility("default"))) uint64_t npu_addmm_rebase_counter(unsigned i)
{
    switch (i) {
        case 0: return accepted.load(std::memory_order_relaxed);
        case 1: return rejected.load(std::memory_order_relaxed);
        case 2: return keys.load(std::memory_order_relaxed);
        case 3: return descriptors.load(std::memory_order_relaxed);
        default: return 0;
    }
}
}  // namespace projection_rebase
