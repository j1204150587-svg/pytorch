#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace op_api {
namespace compact_div {
enum Counter { SCALAR_HIT, TENSOR_HIT, FALLBACK, CONSTANTS, COUNT };
static std::atomic<uint64_t> counts[COUNT]{};

static bool enabled()
{
    static const bool value = [] {
        const char *flag = std::getenv("NPU_HSTU_COMPACT_DIV");
        if (!flag || std::strcmp(flag, "1") != 0) {
            return false;
        }
        std::atexit([] {
            std::fprintf(stderr, "HSTU_COMPACT_DIV scalar_hit=%llu tensor_hit=%llu fallback=%llu constant_init=%llu\n",
                static_cast<unsigned long long>(counts[SCALAR_HIT].load()),
                static_cast<unsigned long long>(counts[TENSOR_HIT].load()),
                static_cast<unsigned long long>(counts[FALLBACK].load()),
                static_cast<unsigned long long>(counts[CONSTANTS].load()));
        });
        return true;
    }();
    return value;
}

static bool eligible(const at::Tensor& x, const at::Tensor& result)
{
    if (x.layout() != at::kStrided || x.dim() != 4 || x.scalar_type() != at::kBFloat16 ||
        x.size(0) < 1 || x.size(0) > 4 || x.size(1) < 12 || x.size(1) > 32 ||
        x.size(2) != 4 || x.size(3) != 64 || x.stride(2) != 0 || x.stride(3) != 1 ||
        x.stride(1) != 64 || (x.size(0) != 1 && x.stride(0) != 64 * x.size(1)) ||
        result.scalar_type() != at::kBFloat16 || x.is_neg() || x.is_conj() ||
        x.has_names() || !at_npu::native::FormatHelper::IsOpInputBaseFormat(x)) {
        return false;
    }
    static const bool available = GetOpApiFuncAddr("aclnnDivGetWorkspaceSize") &&
        GetOpApiFuncAddr("aclnnDiv") && c10_npu::option::OptionsManager::GetTaskQueueEnable() == 1;
    return available && c10_npu::currentStreamCaptureStatus() == c10_npu::CaptureStatus::None;
}

static bool run(const at::Tensor& x, const at::Scalar& scalar, at::Tensor& result, Counter path)
{
    if (!(scalar.isIntegral(false) || scalar.isFloatingPoint()) || scalar.toDouble() != 4.0 ||
        !eligible(x, result)) {
        counts[FALLBACK].fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    // One four-element constant per producer, replaced when that thread switches stream or device.
    // It is allocated and always consumed on its owning stream.
    struct Slot {
        at::Tensor value;
        c10::DeviceIndex device = -1;
        c10::StreamId stream = 0;
    };
    static thread_local Slot slot;
    const auto stream = c10_npu::getCurrentNPUStream(x.device().index());
    if (!slot.value.defined() || slot.device != stream.device_index() || slot.stream != stream.id()) {
        slot.value = at::full({1, 1, 4, 1}, 4, x.options());
        slot.device = stream.device_index();
        slot.stream = stream.id();
        counts[CONSTANTS].fetch_add(1, std::memory_order_relaxed);
    }
    // Discard the expanded axis while preserving the byte location of every input value.
    // The binary ACLNN kernel expands this singleton while it produces its dense fresh result.
    const int64_t b = x.size(0), l = x.size(1);
    const auto compact = x.as_strided({b, l, 1, 64}, {l * 64, 64, 64, 1}, x.storage_offset());
    const auto& divisor = slot.value;
    EXEC_NPU_CMD(aclnnDiv, compact, divisor, result);
    counts[path].fetch_add(1, std::memory_order_relaxed);
    return true;
}

extern "C" __attribute__((visibility("default"))) uint64_t npu_hstu_compact_div_counter(unsigned i)
{
    return i < COUNT ? counts[i].load(std::memory_order_relaxed) : 0;
}
}  // namespace compact_div
}  // namespace op_api
