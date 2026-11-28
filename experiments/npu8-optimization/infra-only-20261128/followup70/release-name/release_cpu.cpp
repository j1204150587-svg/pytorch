#include "torch_npu/csrc/framework/OpParamMaker.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <memory>
#include <pthread.h>
#include <sched.h>
#include <thread>

using Payload = at_npu::native::ExecuteParasOpApi;
void moveOnly(Payload &, Payload &);
constexpr size_t header = sizeof(c10_npu::queue::QueueParas);
constexpr size_t stride = header + at_npu::native::MAX_PARAS_BYTE_SIZE;
static_assert(stride % alignof(Payload) == 0 && header % alignof(Payload) == 0);
struct Ring {
    std::vector<std::max_align_t> raw;
    const size_t n;
    explicit Ring(size_t cap) : raw((cap*stride+sizeof(std::max_align_t)-1)/sizeof(std::max_align_t)), n(cap) {
        for (size_t i=0;i<n;++i) ::new(pointer(i)) Payload();
    }
    void *pointer(size_t i) { return reinterpret_cast<char *>(raw.data())+i*stride+header; }
    Payload &operator[](size_t i) { return *static_cast<Payload *>(pointer(i)); }
    ~Ring() { for(size_t i=0;i<n;++i) (*this)[i].~Payload(); }
};
void pin(int cpu) {
    cpu_set_t a;
    CPU_ZERO(&a); CPU_SET(cpu,&a);
    if (pthread_setaffinity_np(pthread_self(),sizeof(a),&a)) std::abort();
}
struct alignas(128) State { std::atomic<unsigned> v{0}; };
uint64_t ns() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
template<bool name> __attribute__((noinline)) uint64_t handoff(Ring &dst,Ring &src) {
    auto begin=ns();
    for (size_t i=0;i<src.n;++i) {
        if (name) dst[i].Copy(src[i]); else moveOnly(dst[i],src[i]);
        src[i].Release();
    }
    return ns()-begin;
}
void run(const std::vector<std::string>& names, size_t cap, unsigned mode) {
    Ring src(cap),dst(cap);
    State ready,done;
    constexpr unsigned waves=68;
    std::thread worker([&](){
        pin(311);
        for(unsigned wave=0;wave<waves;++wave) {
            while(ready.v.load(std::memory_order_acquire)!=wave+1) asm volatile("yield");
            for(size_t i=0;i<cap;++i) {
                if (!dst[i].customHandler || dst[i].customHandler()!=37) std::abort();
                dst[i].Release();
            }
            done.v.store(wave+1,std::memory_order_release);
        }
    });
    pin(310);
    auto ref=std::make_shared<int>(37);
    for(unsigned wave=0;wave<waves;++wave) {
        const bool copy=std::array<bool,4>{true,false,false,true}[wave%4];
        for(size_t i=0;i<cap;++i) {
            auto &s=src[i];
            const auto &n=names[(wave*cap+i)%names.size()];
            std::memset(s.opType,0,sizeof(s.opType));
            std::memcpy(s.opType,n.data(),std::min(n.size(),sizeof(s.opType)-1));
            if(mode==0) s.customHandler=[](){return 37;};
            else s.customHandler=[ref](){return *ref;};
        }
        const auto elapsed=copy?handoff<true>(dst,src):handoff<false>(dst,src);
        for(size_t i=0;i<cap;++i) if (src[i].customHandler) std::abort();
        ready.v.store(wave+1,std::memory_order_release);
        while(done.v.load(std::memory_order_acquire)!=wave+1) asm volatile("yield");
        if(ref.use_count()!=1) std::abort();
        if(wave>=4) std::printf("%zu,%zu,%zu,%u,%u,%c,%.6f\n",cap,stride,sizeof(Payload),mode,wave,copy?'A':'B',double(elapsed)/cap);
    }
    worker.join();
}
int main(int argc,char **argv) {
    if(argc!=2) return 2;
    std::ifstream f(argv[1]);
    std::vector<std::string> names;
    for(std::string s;std::getline(f,s);) if(!s.empty()) names.push_back(s);
    if(names.empty()) return 3;
    std::puts("slots,stride_bytes,payload_bytes,heap_capture,wave,arm,ns_per_release_handoff");
    for(size_t cap:{size_t(64),size_t(8192)}) for(unsigned mode:{0U,1U}) run(names,cap,mode);
}
