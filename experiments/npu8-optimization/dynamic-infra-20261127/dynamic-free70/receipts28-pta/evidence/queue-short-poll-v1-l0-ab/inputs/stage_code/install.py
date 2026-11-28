"""Patch the pinned Mode 1 consumer's first empty-queue observation."""
from pathlib import Path
import sys


SOURCE = "torch_npu/csrc/core/npu/NPUQueue.cpp"


def replace(src, old, new):
    if src.count(old) != 1:
        raise ValueError(f"unexpected queue source at {old!r}")
    return src.replace(old, new)


def main(root):
    path = root / SOURCE
    src = path.read_text()
    src = replace(src, "#include <sys/eventfd.h>", """#include <sys/eventfd.h>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>

namespace {
enum QueueCounter { Empty, Poll, Hit, Sleep, Notify, CounterCount };
std::atomic<uint64_t> short_poll_counts[CounterCount]{};
void bump(QueueCounter key)
{
    short_poll_counts[key].fetch_add(1, std::memory_order_relaxed);
}
bool short_poll_enabled()
{
    static const bool on = [] {
        const char* value = std::getenv("NPU_DYNAMIC_SHORT_POLL");
        return value && std::strcmp(value, "1") == 0;
    }();
    return on;
}
} // namespace

extern "C" __attribute__((visibility("default"))) uint64_t npu_hstu_short_poll_counter(unsigned key)
{
    return key < CounterCount ? short_poll_counts[key].load(std::memory_order_relaxed) : 0;
}""")
    start = src.index("bool Repository::ReadQueue()")
    end = src.index("    __sync_synchronize();", start)
    prefix, body, suffix = src[:start], src[start:end], src[end:]
    old = """        } else {
            return false;
        }
"""
    new = """        } else {
            bump(Empty);
            if (task_queue_enable != 1 || !short_poll_enabled()) {
                return false;
            }
            bump(Poll);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::microseconds(32);
            // Use the same empty predicate and subsequent publication fence as stock.
            // Recheck the clock in small batches so a sparse producer can still sleep.
            bool ready = false;
            do {
                for (unsigned i = 0; i < 32; ++i) {
                    if (!IsEmptyQueue()) {
                        ready = true;
                        break;
                    }
                }
            } while (!ready && std::chrono::steady_clock::now() < deadline);
            if (!ready) {
                return false;
            }
            bump(Hit);
        }
"""
    src = prefix + replace(body, old, new) + suffix
    src = replace(src, "                s = eventfd_read(efd_read, &u);",
                  "                bump(Sleep);\n                s = eventfd_read(efd_read, &u);")
    src = replace(src, "            s = eventfd_write(efd_read, u);",
                  "            bump(Notify);\n            s = eventfd_write(efd_read, u);")
    path.write_text(src)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
