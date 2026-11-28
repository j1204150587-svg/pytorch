#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
bool viewFlag(const char* name)
{
    const auto* value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

bool viewInlineDimsEnabled()
{
    static const bool enabled = viewFlag("TORCH_NPU_VIEW_INLINE_DIMS");
    return enabled;
}

bool viewInlineDimsAudit()
{
    static const bool audit = viewFlag("TORCH_NPU_VIEW_INLINE_DIMS_AUDIT");
    return audit;
}

std::atomic<uint64_t> inlineDimsCalls{0}, inlineDimsEligible{0}, inlineDimsUsed{0};

void reportInlineDims()
{
    std::fprintf(stderr, "PTA_VIEW_INLINE_DIMS enabled=%d calls=%" PRIu64
        " eligible=%" PRIu64 " used=%" PRIu64 "\n", static_cast<int>(viewInlineDimsEnabled()),
        inlineDimsCalls.load(), inlineDimsEligible.load(), inlineDimsUsed.load());
}

void recordInlineDims(bool active, bool eligible)
{
    static const int registered = std::atexit(reportInlineDims);
    (void)registered;
    inlineDimsCalls.fetch_add(1, std::memory_order_relaxed);
    if (eligible) inlineDimsEligible.fetch_add(1, std::memory_order_relaxed);
    if (active) inlineDimsUsed.fetch_add(1, std::memory_order_relaxed);
}
} // namespace
