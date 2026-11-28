// Count allocations in the actual libtorch shape/stride APIs used by the two branches.
#include <ATen/InferSize.h>
#include <ATen/TensorUtils.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <new>

namespace {
thread_local bool track = false;
thread_local size_t allocations = 0;
volatile int64_t witness = 0;
}

void* operator new(std::size_t size)
{
    if (track) ++allocations;
    if (auto* p=std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}

void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }

int main()
{
    size_t before=0, after=0, cases=0;
    for (int64_t batch=1; batch<=4; ++batch) {
        for (int64_t seq=12; seq<=32; ++seq) {
            const int64_t n=batch*seq;
            c10::IntArrayRef flat(&n,1);
            const int64_t ones[]={1};
            const int64_t target[]={batch,-1,1};
            allocations=0;
            track=true;
            {
                auto sizes=at::infer_size(target,n);
                const auto strides=at::detail::computeStride(flat,ones,sizes);
                auto copy=*strides;
                witness+=sizes[0]+copy[1];
            }
            track=false;
            before+=allocations;
            allocations=0;
            track=true;
            {
                auto sizes=at::infer_size_dv(target,n);
                const auto strides=at::detail::computeStride(flat,ones,sizes);
                witness+=sizes[0]+(*strides)[1];
            }
            track=false;
            after+=allocations;
            ++cases;
        }
    }
    std::printf("{\"cases\":%zu,\"vector_new\":%zu,\"dimvector_new\":%zu,\"witness\":%ld}\n",
                cases,before,after,static_cast<long>(witness));
    if (before!=3*cases || after!=0) return 1;
}
