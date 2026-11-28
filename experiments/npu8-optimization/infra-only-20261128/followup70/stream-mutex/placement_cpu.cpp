// Exercise libc's normal mutexes at the two shipping Stream offsets. The two
// experiment threads communicate on other lines after every batch, so each
// first lock sees a completed recycle write. This is a forced handoff microtest.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <new>
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <vector>

static void pin(int cpu) {
    cpu_set_t cpus;
    CPU_ZERO(&cpus);
    CPU_SET(cpu, &cpus);
    if (pthread_setaffinity_np(pthread_self(),sizeof(cpus),&cpus)) std::abort();
}

static uint64_t counter() {
    uint64_t value;
    asm volatile("isb\n mrs %0,cntvct_el0" : "=r"(value) :: "memory");
    return value;
}

struct alignas(128) Flag { std::atomic<unsigned> value{}; };

static void run(unsigned index,int submit_cpu,int recycle_cpu,unsigned mod,unsigned ratio) {
    constexpr unsigned rounds=9000;
    alignas(128) std::array<std::byte,512> storage{};
    static_assert(sizeof(std::mutex)==48);
    auto *submit=new(storage.data()+mod+40) std::mutex;
    auto *recycle=new(storage.data()+mod+88) std::mutex;
    Flag request,done;
    pin(submit_cpu);
    std::thread worker([&]() {
        pin(recycle_cpu);
        for (unsigned i=1;i<=rounds;++i) {
            while(request.value.load(std::memory_order_acquire)!=i) asm volatile("yield");
            recycle->lock();
            recycle->unlock();
            done.value.store(i,std::memory_order_release);
        }
    });
    std::vector<uint64_t> first,batch;
    first.reserve(rounds); batch.reserve(rounds);
    for(unsigned i=1;i<=rounds;++i) {
        request.value.store(i,std::memory_order_release);
        while(done.value.load(std::memory_order_acquire)!=i) asm volatile("yield");
        const auto start=counter();
        submit->lock();
        submit->unlock();
        const auto end_first=counter();
        for(unsigned j=1;j<ratio;++j) {
            submit->lock();
            submit->unlock();
        }
        const auto finish=counter();
        if (i>500) {
            first.push_back(end_first-start); batch.push_back(finish-start);
        }
    }
    worker.join();
    submit->~mutex();
    recycle->~mutex();
    uint64_t freq;
    asm volatile("mrs %0,cntfrq_el0" : "=r"(freq));
    std::sort(first.begin(),first.end()); std::sort(batch.begin(),batch.end());
    printf("%u,%d,%d,%u,%u,%zu,%llu",index,submit_cpu,recycle_cpu,mod,ratio,first.size(),
           static_cast<unsigned long long>(freq));
    for(auto v:{&first,&batch}) for(double q:{.5,.95,.99}) {
        const uint64_t ticks=v->at(static_cast<size_t>(q*(v->size()-1)));
        printf(",%.2f",ticks*1.e9/freq);
    }
    puts(""); fflush(stdout);
}

int main() {
    puts("index,submit_cpu,recycle_cpu,base_mod128,submit_per_recycle,n,clock_hz,first_p50_ns,first_p95_ns,first_p99_ns,batch_p50_ns,batch_p95_ns,batch_p99_ns");
    unsigned index=0;
    for(auto cpus:{std::array<int,2>{188,189},std::array<int,2>{266,269}}) {
        unsigned ratio=cpus[0]==188 ? 36 : 54;
        // The shipping hot allocation in both ranks had remainder 96.
        for(unsigned mod:{96,64,64,96,64,96,96,64,96,64,64,96})
            run(index++,cpus[0],cpus[1],mod,ratio);
    }
}
