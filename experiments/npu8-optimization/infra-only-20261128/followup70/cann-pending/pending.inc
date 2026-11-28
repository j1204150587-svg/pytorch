#if defined(__aarch64__) && defined(__linux__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
namespace pending_probe {
bool flag(const char *name)
{
    const char *value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}
enum class Instruction { llsc, lse };
const bool auditing = flag("NPU_CANN_PENDING_COUNT");
const bool hasLse = (getauxval(AT_HWCAP) & HWCAP_ATOMICS) != 0;
const bool profiling = flag("NPU_CANN_PENDING_PROFILE");
std::atomic<Instruction> mode {(hasLse && flag("NPU_CANN_PENDING_LSE")) ? Instruction::lse : Instruction::llsc};
std::array<std::atomic<uint64_t>, 2> counts {};
template <class Counter>
inline void Increment(Counter &counter)
{
    // C++17 [basic.compound]: a standard-layout object and its first non-static
    // member are pointer-interconvertible. Atomic<uint32_t> has only val_.
    static_assert(std::is_standard_layout<Counter>::value && sizeof(Counter) == sizeof(std::atomic<uint32_t>) &&
                  alignof(Counter) == alignof(std::atomic<uint32_t>) && std::atomic<uint32_t>::is_always_lock_free);
    auto &cell = *reinterpret_cast<std::atomic<uint32_t> *>(&counter);
    uint32_t prior;
    const auto selected = mode.load(std::memory_order_relaxed);
    if (selected == Instruction::lse) {
        const uint32_t step = 1;
        // An actual destination register is required for the acquire half.
        asm volatile(".arch_extension lse\n\tldaddal %w[step], %w[prior], %[cell]"
                     : [prior] "=&r"(prior), [cell] "+Q"(cell) : [step] "r"(step) : "memory");
    } else {
        uint32_t status;
        asm volatile("1: ldaxr %w[prior], %[cell]\n\tadd %w[prior], %w[prior], #1\n\t"
                     "stlxr %w[status], %w[prior], %[cell]\n\tcbnz %w[status], 1b"
                     : [prior] "=&r"(prior), [status] "=&r"(status), [cell] "+Q"(cell) : : "memory");
    }
    if (auditing) counts[static_cast<unsigned>(selected)].fetch_add(1, std::memory_order_relaxed);
}
} // namespace pending_probe
extern "C" __attribute__((visibility("default"))) uint64_t npu_cann_pending_count(unsigned int field)
{
    if (field == 2) return pending_probe::hasLse;
    if (field == 3) return static_cast<unsigned>(pending_probe::mode.load(std::memory_order_relaxed));
    return field < 2 ? pending_probe::counts[field].load(std::memory_order_relaxed) : 0;
}
extern "C" __attribute__((visibility("default"))) void npu_cann_pending_profile_select(bool enabled)
{
    if (pending_probe::profiling) {
        pending_probe::mode.store(enabled && pending_probe::hasLse ? pending_probe::Instruction::lse :
                                  pending_probe::Instruction::llsc,std::memory_order_relaxed);
    }
}
#endif
