#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"

namespace c10_npu::NPUCachingAllocator {
bool ownedByCopyStream(const c10::DataPtr&, aclrtStream);
}

namespace {
std::array<std::atomic<uint64_t>,4> consumer_copy_count{};

bool consumer_copy_flag(const char* name)
{
    const char* value = std::getenv(name);
    return value && std::strcmp(value,"1") == 0;
}

void count_consumer_copy(size_t index)
{
    static const bool enabled = consumer_copy_flag("NPU_DYNAMIC_REBASE_COUNT");
    if (enabled) consumer_copy_count[index].fetch_add(1, std::memory_order_relaxed);
}

bool use_consumer_copy(const at::Tensor& dst, const at::Tensor& src)
{
    static const bool enabled = consumer_copy_flag("NPU_INPLACE_COPY_CONSUMER");
    if (!enabled) return false;
    bool eligible = c10_npu::option::OptionsManager::GetTaskQueueEnable() == 1 &&
        dst.scalar_type() == at::kBFloat16 && src.scalar_type() == at::kBFloat16 &&
        dst.dim() == 1 && src.dim() == 1 && dst.size(0) == 4096 && src.size(0) == 4096 &&
        dst.stride(0) == 1 && src.stride(0) == 1 &&
        !dst.is_conj() && !src.is_conj() && !dst.is_neg() && !src.is_neg() &&
        dst.storage().unsafeGetStorageImpl() != src.storage().unsafeGetStorageImpl() &&
        at_npu::native::FormatHelper::GetFormat(dst) == ACL_FORMAT_ND &&
        at_npu::native::FormatHelper::GetFormat(src) == ACL_FORMAT_ND;
    if (eligible) {
        const auto stream = c10_npu::getCurrentNPUStream();
        eligible = !stream.isSyncLaunchStream() &&
            c10_npu::currentStreamCaptureStatusMayInitCtx() != c10_npu::CaptureStatus::Active &&
            c10_npu::NPUCachingAllocator::ownedByCopyStream(dst.storage().data_ptr(), stream.stream(false)) &&
            c10_npu::NPUCachingAllocator::ownedByCopyStream(src.storage().data_ptr(), stream.stream(false));
    }
    count_consumer_copy(eligible ? 0 : 1);
    return eligible;
}
}

extern "C" __attribute__((visibility("default"))) uint64_t npu_inplace_copy_rebase_counter(unsigned index)
{
    return index < consumer_copy_count.size() ? consumer_copy_count[index].load(std::memory_order_relaxed) : 0;
}
