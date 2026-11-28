// CPU-only full-queue protocol test. The wait counter and registration are from the PTA patch.
#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <sys/eventfd.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace c10_npu::option {
struct OptionsManager {
    static int GetTaskQueueEnable() { return 1; }
    static bool GetPerStreamQueue() { return false; }
};
}
#include "waiters.inc"

static void require(bool ok, const char* msg)
{
    if (!ok) { std::cerr << msg << std::endl; std::abort(); }
}

struct Ring {
    static constexpr unsigned size = 4;
    std::array<unsigned, size> slots{};
    std::atomic<unsigned> read{0}, write{0}, blocked{0}, interrupted{0};
    std::atomic<bool> stopping{false};
    ProducerWakeState state;
    std::mutex producer;
    int fd = eventfd(0, EFD_SEMAPHORE);

    ~Ring() { close(fd); }
    bool full() { return (write.load() + 1) % size == read.load(); }
    void notify(uint64_t count)
    {
        int rc;
        do { rc = eventfd_write(fd, count); } while (rc < 0 && errno == EINTR);
        require(rc == 0, "eventfd write failed");
    }
    bool push(unsigned n)
    {
        while (!stopping.load()) {
            {
                std::lock_guard<std::mutex> lock(producer);
                if (stopping.load()) return false;
                if (!full()) {
                    unsigned i = write.load();
                    slots[i] = n;
                    write.store((i + 1) % size);
                    return true;
                }
            }
            FullWaitRegistration registered(state);
            std::atomic_thread_fence(std::memory_order_seq_cst);
            if (!full() || stopping.load()) continue;
            blocked.fetch_add(1);
            uint64_t value = 0;
            int rc = eventfd_read(fd, &value);
            blocked.fetch_sub(1);
            if (rc < 0 && errno == EINTR) { interrupted.fetch_add(1); continue; }
            require(rc == 0 && value == 1, "expected one wake credit");
        }
        return false;
    }
    bool pop(unsigned& value)
    {
        unsigned i = read.load();
        if (i == write.load()) return false;
        value = slots[i];
        read.store((i + 1) % size);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        state.drained.fetch_add(1);
        if (state.waiters.load(std::memory_order_seq_cst)) {
            notify(1);
            state.normalWrites.fetch_add(1);
        }
        return true;
    }
    void clear()
    {
        stopping.store(true);
        read.store(write.load());
        std::atomic_thread_fence(std::memory_order_seq_cst);
        notify(std::max(1u, state.waiters.load(std::memory_order_seq_cst)));
    }
};

int main()
{
    struct sigaction action{};
    action.sa_handler = [](int) {};
    sigaction(SIGUSR1, &action, nullptr);
    constexpr unsigned nthreads = 8, perThread = 1500;
    Ring ring;
    std::vector<std::thread> producers;
    std::vector<unsigned> seen(nthreads * perThread);
    for (unsigned i = 0; i < nthreads; ++i) {
        producers.emplace_back([&, i] {
            for (unsigned j = 0; j < perThread; ++j)
                require(ring.push(i * perThread + j), "unexpected shutdown");
        });
    }
    while (ring.blocked.load() != nthreads) std::this_thread::yield();
    while (!ring.interrupted.load()) {
        for (auto& t : producers) pthread_kill(t.native_handle(), SIGUSR1);
        usleep(1000);
    }
    for (unsigned count = 0; count < seen.size();) {
        unsigned x;
        if (!ring.pop(x)) { std::this_thread::yield(); continue; }
        require(x < seen.size() && !seen[x]++, "duplicate or corrupt value");
        ++count;
    }
    for (auto& t : producers) t.join();
    require(!ring.state.waiters.load(), "leaked waiter");
    Ring empty;
    unsigned value = 0;
    for (unsigned i = 0; i < 10000; ++i) require(empty.push(i) && empty.pop(value) && value == i, "idle mismatch");
    require(empty.state.normalWrites.load() == 0, "unexpected idle notification");

    Ring closing;
    for (unsigned i = 0; i < Ring::size - 1; ++i) require(closing.push(i), "cannot prefill");
    producers.clear();
    for (unsigned i = 0; i < nthreads; ++i)
        producers.emplace_back([&, i] { require(!closing.push(i), "producer escaped shutdown"); });
    while (closing.blocked.load() != nthreads) std::this_thread::yield();
    closing.clear();
    for (auto& t : producers) t.join();
    require(!closing.state.waiters.load(), "stale shutdown waiter");
    std::cout << "{\"items\":" << seen.size() << ",\"producers\":" << nthreads
              << ",\"eintr\":" << ring.interrupted << ",\"full_registrations\":" << ring.state.registrations
              << ",\"normal_writes\":" << ring.state.normalWrites << ",\"idle_items\":10000,\"idle_writes\":0"
              << ",\"shutdown_waiters_released\":" << nthreads << "}" << std::endl;
}
