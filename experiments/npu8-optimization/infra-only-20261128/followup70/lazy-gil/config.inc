#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
std::atomic<bool> pythonReady{false};
std::atomic<uint64_t> lazyCalls{0}, lazyHits{0}, lazyWaitNs{0}, lazyMaxNs{0};

bool envOn(const char* name)
{
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

bool lazyEnabled()
{
    static const bool enabled = envOn("TORCH_NPU_LAZY_GIL_FAST");
    return enabled;
}

bool pythonRuntimeLive()
{
#if PY_VERSION_HEX >= 0x030D0000
    return Py_IsInitialized() && !Py_IsFinalizing();
#else
    return Py_IsInitialized() && !_Py_IsFinalizing();
#endif
}

void reportLazy()
{
    std::fprintf(stderr, "PTA_LAZY_GIL enabled=%d calls=%" PRIu64 " fast=%" PRIu64
                 " acquire_ns=%" PRIu64 " max_ns=%" PRIu64 "\n", static_cast<int>(lazyEnabled()),
                 lazyCalls.load(), lazyHits.load(), lazyWaitNs.load(), lazyMaxNs.load());
}

bool lazyAudit()
{
    static const bool enabled = [] {
        if (!envOn("TORCH_NPU_LAZY_GIL_AUDIT")) return false;
        std::atexit(reportLazy);
        return true;
    }();
    return enabled;
}

void countLazy(std::atomic<uint64_t>& count)
{
    count.fetch_add(1, std::memory_order_relaxed);
}
} // namespace
