#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "torch_npu/csrc/core/npu/NPUMacros.h"

namespace {
struct alignas(64) ProducerWakeState {
    std::atomic<uint32_t> waiters{0};
    std::atomic<uint64_t> registrations{0}, fullReads{0};
    std::atomic<uint64_t> drained{0}, normalWrites{0}, writesWithoutWaiter{0};
    std::atomic<uint64_t> clearWrites{0};
};
ProducerWakeState wakeStates[C10_COMPILE_TIME_MAX_NPUS];

bool auditProducerWake()
{
    static const bool audit = [] {
        const char* value = std::getenv("TORCH_NPU_QUEUE_WAKE_AUDIT");
        return value && std::strcmp(value, "1") == 0;
    }();
    return audit;
}

bool scopedProducerWake()
{
    static const bool enabled = [] {
        const char* value = std::getenv("TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY");
        return value && std::strcmp(value, "1") == 0 &&
               c10_npu::option::OptionsManager::GetTaskQueueEnable() == 1 &&
               !c10_npu::option::OptionsManager::GetPerStreamQueue();
    }();
    return enabled;
}

void reportProducerWakes()
{
    for (unsigned i = 0; i < C10_COMPILE_TIME_MAX_NPUS; ++i) {
        auto& s = wakeStates[i];
        if (s.drained.load() == 0) {
            continue;
        }
        std::fprintf(stderr,
            "PTA_QUEUE_PRODUCER_WAKE device=%u enabled=%d drained=%" PRIu64
            " registrations=%" PRIu64 " full_reads=%" PRIu64 " writes=%" PRIu64
            " writes_without_observed_waiter=%" PRIu64 " clear_writes=%" PRIu64 " waiters=%u\n",
            i, static_cast<int>(scopedProducerWake()), s.drained.load(), s.registrations.load(),
            s.fullReads.load(), s.normalWrites.load(), s.writesWithoutWaiter.load(),
            s.clearWrites.load(), s.waiters.load());
    }
}

void installProducerWakeReport()
{
    static const int registered = std::atexit(reportProducerWakes);
    (void)registered;
}

class FullWaitRegistration {
public:
    explicit FullWaitRegistration(ProducerWakeState& s) : state(s),
        interested(scopedProducerWake() || auditProducerWake())
    {
        // Publish interest before the producer rechecks queue capacity.
        if (interested) state.waiters.fetch_add(1, std::memory_order_seq_cst);
        if (auditProducerWake()) state.registrations.fetch_add(1, std::memory_order_relaxed);
    }
    ~FullWaitRegistration()
    {
        if (interested) state.waiters.fetch_sub(1, std::memory_order_seq_cst);
    }
    FullWaitRegistration(const FullWaitRegistration&) = delete;
    FullWaitRegistration& operator=(const FullWaitRegistration&) = delete;
private:
    ProducerWakeState& state;
    const bool interested;
};
}  // namespace
