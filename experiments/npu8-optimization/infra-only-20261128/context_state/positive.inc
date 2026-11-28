namespace {
std::atomic<uint64_t> active_context_mask{0};
// Mutated only while holding mtx. A vendor reset may call back on the same thread.
unsigned context_reset_depth = 0;
std::atomic<uint64_t> context_read_counts[4]{};

bool context_flag(const char* name)
{
    const char* value = std::getenv(name);
    return value && std::strcmp(value, "1") == 0;
}

uint64_t context_bit(int32_t device)
{
    static const bool enabled = context_flag("NPU_CONTEXT_POSITIVE_MASK");
    if (!enabled || static_cast<uint32_t>(device) >= 64) {
        return 0;
    }
    return uint64_t{1} << static_cast<unsigned>(device);
}

void count_context_read(unsigned index)
{
    static const bool enabled = context_flag("NPU_DYNAMIC_REBASE_COUNT");
    if (enabled) context_read_counts[index].fetch_add(1, std::memory_order_relaxed);
}

struct ActiveContextReset {
    ActiveContextReset()
    {
        ++context_reset_depth;
        active_context_mask.exchange(0, std::memory_order_acq_rel);
    }
    ~ActiveContextReset() { --context_reset_depth; }
};
} // namespace

extern "C" __attribute__((visibility("default"))) uint64_t npu_active_context_counter(unsigned index)
{
    return index < 4 ? context_read_counts[index].load(std::memory_order_relaxed) : 0;
}
