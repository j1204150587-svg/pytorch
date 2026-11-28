#include "pending.inc"
#include <cstdio>
#include <thread>
#include <vector>
class Counter {
public:
    uint32_t Value() {return value_.load();}
private:
    std::atomic<uint32_t> value_ {0};
};
int main()
{
    Counter counter;
    constexpr unsigned workers=8, loops=200000;
    std::vector<std::thread> pool;
    for (unsigned i=0;i<workers;++i)
        pool.emplace_back([&](){for (unsigned j=0;j<loops;++j) pending_probe::Increment(counter);});
    for (auto& thread:pool) thread.join();
    std::printf("{\"counter\":%u,\"expected\":%u,\"hwcap_atomics\":%llu,\"mode\":%llu,\"llsc\":%llu,\"lse\":%llu}\n",
                counter.Value(),workers*loops,(unsigned long long)npu_cann_pending_count(2),
                (unsigned long long)npu_cann_pending_count(3),(unsigned long long)npu_cann_pending_count(0),
                (unsigned long long)npu_cann_pending_count(1));
    const auto selected=npu_cann_pending_count(3);
    return counter.Value()!=workers*loops || npu_cann_pending_count(selected)!=workers*loops ||
           npu_cann_pending_count(1-selected)!=0;
}
