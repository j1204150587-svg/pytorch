#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "torch_npu/csrc/core/NPUBridge.h"

namespace div_geometry {
enum Counts { Ineligible, FirstKey, Reused, Checked, ExecHit, ExecMiss, NumCounts };
static std::atomic<uint64_t> counts[NumCounts]{};

static bool flag(const char *name)
{
    const char *s = std::getenv(name);
    return s && std::strcmp(s,"1") == 0;
}
static const bool enabled = flag("NPU_DIV_GEOMETRY_KEY");
static const bool audit = flag("NPU_DIV_GEOMETRY_AUDIT");
static const bool counting = audit || flag("NPU_DIV_GEOMETRY_COUNT");
static void count(Counts c)
{
    if (counting) {
        counts[c].fetch_add(1, std::memory_order_relaxed);
    }
}
struct Report {
    ~Report()
    {
        if (enabled && counting) {
            std::fprintf(stderr, "DIV_GEOMETRY fallback=%llu first=%llu reuse=%llu checked=%llu exec_hit=%llu exec_miss=%llu\n",
                (unsigned long long)counts[Ineligible].load(), (unsigned long long)counts[FirstKey].load(),
                (unsigned long long)counts[Reused].load(), (unsigned long long)counts[Checked].load(),
                (unsigned long long)counts[ExecHit].load(), (unsigned long long)counts[ExecMiss].load());
        }
    }
};
static Report report;
extern "C" __attribute__((visibility("default"))) uint64_t npu_div_geometry_count(unsigned i)
{
    return i < NumCounts ? counts[i].load(std::memory_order_relaxed) : 0;
}

// All unlisted tensor/scalar fields consumed by stock add_param_to_buf are
// checked against constants in geometry(). The addresses are rebound per call.
using Key = std::array<uint64_t,12>;
struct Slot {
    bool valid = false;
    Key key{};
    uint64_t hash = 0;
};
static thread_local Slot slot;

static int format(const at::Tensor& t)
{
    auto *s = t.storage().unsafeGetStorageImpl();
    return typeid(*s) == typeid(torch_npu::NPUStorageImpl) ?
        static_cast<int>(torch_npu::NPUBridge::GetNpuStorageImpl(t)->npu_desc_.npu_format_) : ACL_FORMAT_ND;
}
static bool geometry(const at::Tensor& x, const at::Scalar& scalar, const at::Tensor& out, Key& key)
{
    if (x.dim() != 4 || out.dim() != 4 || x.scalar_type() != at::kBFloat16 || out.scalar_type() != at::kBFloat16 ||
        x.device().type() != c10::DeviceType::PrivateUse1 || out.device() != x.device() ||
        x.layout() != at::kStrided || out.layout() != at::kStrided ||
        !(scalar.type() == at::kLong || scalar.type() == at::kDouble) || scalar.toDouble() != 4.0) {
        return false;
    }
    int64_t b = x.size(0), l = x.size(1);
    if (b < 1 || b > 4 || l < 12 || l > 32 || x.size(2) != 4 || x.size(3) != 64 ||
        !x.sizes().equals(out.sizes()) || x.stride(1) != 64 || x.stride(2) != 0 || x.stride(3) != 1 ||
        out.stride(0) != l * 256 || out.stride(1) != 256 || out.stride(2) != 64 || out.stride(3) != 1 ||
        !x.unsafeGetTensorImpl()->storage_initialized() || !out.unsafeGetTensorImpl()->storage_initialized()) {
        return false;
    }
    key = {uint64_t(b), uint64_t(l), uint64_t(x.stride(0)), uint64_t(x.storage_offset()),
           uint64_t(out.storage_offset()), x.storage().nbytes(), out.storage().nbytes(),
           uint64_t(format(x)), uint64_t(format(out)), uint64_t(scalar.type()), 0, 0};
    return true;
}
}  // namespace div_geometry
