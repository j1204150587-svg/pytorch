// Exercise the real NPUStorageImpl constructor with CPU-owned, zero-sized storage metadata.
#include "torch_npu/csrc/core/NPUStorageImpl.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <thread>
#include <vector>
#include <sched.h>

int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    const int threads=std::atoi(argv[1]);
    const int count=std::atoi(argv[2]);
    if (threads<1 || threads>8 || count<1) return 2;
    const char* mode=std::getenv("TORCH_NPU_STORAGE_ATOMIC_ID");
    if (!mode || (mode[0]!='1' && threads>1)) return 3;
    cpu_set_t allowed;
    if (sched_getaffinity(0,sizeof(allowed),&allowed)) return 4;
    std::vector<int> cpus;
    for (int i=0;i<CPU_SETSIZE;++i) if (CPU_ISSET(i,&allowed)) cpus.push_back(i);
    if (cpus.size()<static_cast<size_t>(threads)) return 4;
    std::vector<uint64_t> ids(static_cast<size_t>(threads)*count);
    std::vector<std::thread> workers;
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> go{false};
    for (int lane=0;lane<threads;++lane) {
        workers.emplace_back([&,lane] {
            cpu_set_t mask;
            CPU_ZERO(&mask);
            CPU_SET(cpus[lane],&mask);
            if (sched_setaffinity(0,sizeof(mask),&mask)) std::terminate();
            ready.fetch_add(1,std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) std::this_thread::yield();
            for (int i=0;i<count;++i) {
                torch_npu::NPUStorageImpl s(c10::StorageImpl::use_byte_size_t(),0,
                    c10::DataPtr(nullptr,c10::Device(c10::DeviceType::CPU)),nullptr,false);
                ids[static_cast<size_t>(lane)*count+i]=s.get_unique_id();
            }
        });
    }
    while (ready.load(std::memory_order_acquire)!=static_cast<uint32_t>(threads)) std::this_thread::yield();
    const auto start=std::chrono::steady_clock::now();
    go.store(true,std::memory_order_release);
    for (auto& t:workers) t.join();
    const auto elapsed=std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now()-start).count();
    std::sort(ids.begin(),ids.end());
    size_t invalid=0;
    for (size_t i=1;i<ids.size();++i) invalid+=ids[i]!=ids[i-1]+1;
    std::printf("{\"enabled\":%d,\"threads\":%d,\"count\":%zu,\"first\":%lu,\"last\":%lu,"
                "\"nonconsecutive\":%zu,\"constructor_ns\":%.3f}\n",mode[0]=='1',threads,ids.size(),
                static_cast<unsigned long>(ids.front()),static_cast<unsigned long>(ids.back()),invalid,
                static_cast<double>(elapsed)/ids.size());
    return invalid!=0;
}
