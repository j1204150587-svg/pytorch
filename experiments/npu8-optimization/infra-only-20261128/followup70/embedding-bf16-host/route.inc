#include <ATen/Context.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "torch_npu/csrc/core/npu/NpuVariables.h"

namespace dynamic_embedding_rebase {
const View* for_tensor(const at::Tensor&);
}

namespace {
struct Bf16EmbeddingEntry {
    void* prepare = nullptr;
    void* run = nullptr;
    Bf16EmbeddingEntry()
    {
        const char* path = std::getenv("NPU_STUDY_BF16_LIBRARY");
        if (!path || !*path) {
            return;
        }
        void* handle = dlopen(path, RTLD_LOCAL | RTLD_NOW);
        if (handle) {
            // Task Queue may execute the callback after the producer returns.
            prepare = dlsym(handle, "aclnnStudyBf16EmbeddingGetWorkspaceSize");
            run = dlsym(handle, "aclnnStudyBf16Embedding");
        }
    }
};

const Bf16EmbeddingEntry& bf16Entry()
{
    static const Bf16EmbeddingEntry entry;
    return entry;
}

void* resolveBf16EmbeddingFunction(const char* name)
{
    if (!std::strcmp(name,"aclnnStudyBf16EmbeddingGetWorkspaceSize")) {
        return bf16Entry().prepare;
    }
    if (!std::strcmp(name,"aclnnStudyBf16Embedding")) {
        return bf16Entry().run;
    }
    return GetOpApiFuncAddr(name);
}

bool selectNativeBf16Embedding(const at::Tensor& grad, const at::Tensor& ids, int64_t weights, int64_t pad, bool scale)
{
    static const bool enabled = []{ const char* s=std::getenv("NPU_EMBED_NATIVE_BF16"); return s && !std::strcmp(s,"1"); }();
    static const bool audit = []{ const char* s=std::getenv("NPU_EMBED_BF16_AUDIT"); return s && !std::strcmp(s,"1"); }();
    const auto& entry = bf16Entry();
    bool use = enabled && entry.prepare && entry.run && dynamic_embedding_rebase::for_tensor(ids) &&
        c10_npu::GetSocVersion()==c10_npu::SocVersion::Ascend910_9362 && !at::globalContext().deterministicAlgorithms() &&
        weights==64 && !scale && (pad==-1 || (pad>=0 && pad<weights)) && grad.scalar_type()==at::kBFloat16 &&
        grad.device()==ids.device() && grad.is_contiguous() && grad.dim()==4 && ids.size(0)<=4 &&
        ids.size(1)>=12 && ids.size(1)<=32 && grad.size(0)==ids.size(0) && grad.size(1)==ids.size(1) &&
        grad.size(2)==4 && grad.size(3)==64;
    if (use) {
        int64_t deterministic=-1;
        use=aclrtGetSysParamOpt(ACL_OPT_DETERMINISTIC,&deterministic)==ACL_SUCCESS && deterministic==0;
    }
    if (audit) {
        struct Tally {
            uint64_t hit=0, fallback=0;
            ~Tally() { std::fprintf(stderr,"NPU_NATIVE_BF16 hit=%llu fallback=%llu\n",
                static_cast<unsigned long long>(hit),static_cast<unsigned long long>(fallback)); }
        };
        static thread_local Tally count;
        if (use) { ++count.hit; } else { ++count.fallback; }
    }
    return use;
}

// The two private symbols are local to this eager operator. All other ACLNN
// cache, command, and workspace helpers go through the shipping resolver.
#define GetOpApiFuncAddr resolveBf16EmbeddingFunction
void runNativeBf16Embedding(const at::Tensor& grad, const at::Tensor& ids, int64_t weights,
                            int64_t pad, bool scale, const at::Tensor& out)
{
    EXEC_NPU_CMD(aclnnStudyBf16Embedding, grad, ids, weights, pad, scale, out);
}
#undef GetOpApiFuncAddr
} // namespace
