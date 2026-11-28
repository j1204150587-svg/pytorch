#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <new>

namespace {
using cce::runtime::Stream;
std::atomic<uint64_t> layoutCounters[6]{};
std::atomic<uintptr_t> observedStreams[128]{};
std::atomic<uintptr_t> observedHandles[128]{};
bool splitRequested() {
    static const bool enabled = [] {
        const char *value = std::getenv("NPU_CANN_STREAM_LAYOUT");
        return value && value[0] == '1' && value[1] == '\0';
    }();
    return enabled;
}
// These checked offsets belong only to this pinned v100 build. The ordinary Stream
// object and its deleting destructor remain exactly the ones shipped by this TU.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
static_assert(offsetof(Stream, streamMutex_) == 0x28 && offsetof(Stream, recycleMutex_) == 0x58,
              "Unexpected v100 Stream mutex offsets");
static_assert(alignof(Stream) <= alignof(std::max_align_t), "Stream needs overaligned storage");
#pragma GCC diagnostic pop
struct ReleaseRawBlock {
    void operator()(void *pointer) const noexcept { ::operator delete(pointer); }
};
using RawBlock = std::unique_ptr<void, ReleaseRawBlock>;
bool splitHotFields(const void *p) {
    const auto addr = reinterpret_cast<uintptr_t>(p);
    return ((addr + 0x28 + 12) >> 7) != ((addr + 0x58) >> 7)
        && ((addr + 0x28 + 12) >> 6) != ((addr + 0x58) >> 6);
}
}
extern "C" __attribute__((visibility("default"))) uint64_t npu_cann_stream_layout_count(unsigned i) {
    return i<6 ? layoutCounters[i].load(std::memory_order_relaxed) : 0;
}
extern "C" __attribute__((visibility("default"))) uintptr_t npu_cann_stream_layout_object(uintptr_t handle) {
    for (unsigned i=0; i<128; ++i) {
        if (handle && observedHandles[i].load(std::memory_order_acquire) == handle)
            return observedStreams[i].load(std::memory_order_relaxed);
    }
    return 0;
}
extern "C" __attribute__((visibility("default"))) unsigned npu_cann_stream_layout_classify(uintptr_t handle) {
    auto addr = npu_cann_stream_layout_object(handle);
    return addr ? (splitHotFields(reinterpret_cast<void *>(addr)) ? 2U : 1U) : 0U;
}
