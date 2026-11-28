#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {
std::atomic<uint64_t> idCalls{0}, idNanos{0}, idMaxNanos{0};
bool storageFlag(const char* name)
{
    const char* v = std::getenv(name);
    return v && std::strcmp(v, "1") == 0;
}

bool atomicIds()
{
    static const bool enabled = storageFlag("TORCH_NPU_STORAGE_ATOMIC_ID");
    return enabled;
}

void reportIds()
{
    std::fprintf(stderr, "PTA_STORAGE_ID enabled=%d calls=%" PRIu64 " elapsed_ns=%" PRIu64
        " max_ns=%" PRIu64 "\n", static_cast<int>(atomicIds()), idCalls.load(), idNanos.load(), idMaxNanos.load());
}

bool auditIds()
{
    static const bool audit = [] {
        if (!storageFlag("TORCH_NPU_STORAGE_ID_AUDIT")) return false;
        std::atexit(reportIds);
        return true;
    }();
    return audit;
}

void countId(std::chrono::steady_clock::time_point start)
{
    const uint64_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-start).count();
    idCalls.fetch_add(1, std::memory_order_relaxed);
    idNanos.fetch_add(ns, std::memory_order_relaxed);
    uint64_t max = idMaxNanos.load(std::memory_order_relaxed);
    while (max < ns && !idMaxNanos.compare_exchange_weak(max,ns,std::memory_order_relaxed)) {}
}
}
