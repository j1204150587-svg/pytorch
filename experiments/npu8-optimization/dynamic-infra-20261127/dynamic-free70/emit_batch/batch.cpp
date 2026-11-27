// Use the public ACLNN API to submit distinct gradients on one caller-owned stream.
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include "aclnn/acl_meta.h"
#include "aclnnop/aclnn_embedding_dense_backward.h"

struct Plan {
    aclOpExecutor* exec = nullptr;
    aclTensor* grad = nullptr;
    aclTensor* ids = nullptr;
    aclTensor* out = nullptr;
    uint64_t bytes = 0;
    bool retained = false;

    void release() {
        if (retained && exec) aclDestroyAclOpExecutor(exec);
        if (grad) aclDestroyTensor(grad);
        if (ids) aclDestroyTensor(ids);
        if (out) aclDestroyTensor(out);
    }
};

struct Batch {
    std::array<Plan, 16> entries{};
    int64_t batch, length, tables;
    uint64_t largest = 0;
    int builds = 0, launches = 0;
    static constexpr int64_t rows = 64, width = 64, features = 4;

    int prepare(Plan& p, int table, char* diff, char* ids_base, char* output) {
        const int64_t residual = (table * features) % 64;
        const int64_t shift = table * features - residual;
        const int64_t gshape[]{batch, length, features, width};
        const int64_t gstride[]{length*features*width, features*width, width, 1};
        const int64_t gd = batch*length*features*width;
        const int64_t ishape[]{batch, length, features};
        const int64_t istride[]{length*tables*features, tables*features, 1};
        const int64_t extent = ((residual + (batch*length-1)*tables*features + features + 63)/64)*64;
        const int64_t oshape[]{rows,width}, ostride[]{width,1}, od=rows*width;
        if (shift+extent > batch*length*tables*features) return -1;
        p.grad=aclCreateTensor(gshape,4,ACL_BF16,gstride,0,ACL_FORMAT_NCHW,&gd,1,diff);
        p.ids=aclCreateTensor(ishape,3,ACL_INT64,istride,residual,ACL_FORMAT_NCL,&extent,1,ids_base+8*shift);
        p.out=aclCreateTensor(oshape,2,ACL_BF16,ostride,0,ACL_FORMAT_ND,&od,1,output+2*table*od);
        if (!p.grad || !p.ids || !p.out) return -2;
        ++builds;
        return aclnnEmbeddingDenseBackwardGetWorkspaceSize(p.grad,p.ids,rows,-1,false,p.out,&p.bytes,&p.exec);
    }
};

extern "C" void* packet_new(int64_t b,int64_t l,int64_t t) {
    if (b<=0 || b>4 || l<=0 || l>32 || t!=1024) return nullptr;
    return new Batch{{},b,l,t};
}
extern "C" int packet_plan(Batch* b,char* grad,char* ids,char* out,uint64_t* bytes) {
    if (!b || !grad || !ids || !out || !bytes) return -1;
    for (int t=0;t<16;++t) {
        auto& plan=b->entries[t];
        int rc=b->prepare(plan,t,grad,ids,out);
        if (rc) return rc;
        rc=aclSetAclOpExecutorRepeatable(plan.exec);
        if (rc) return rc;
        plan.retained=true;
        b->largest=std::max(b->largest,plan.bytes);
    }
    *bytes=b->largest;
    return 0;
}
extern "C" int packet_submit(Batch* b,char* grad,char* ids,char* out,void* workspace,aclrtStream stream) {
    if (!b || !workspace) return -1;
    for (int t=0;t<b->tables;++t) {
        auto& plan=b->entries[t%16];
        int rc=aclSetInputTensorAddr(plan.exec,0,plan.grad,grad);
        if (rc) return rc;
        rc=aclSetInputTensorAddr(plan.exec,1,plan.ids,ids+(t/16)*512);
        if (rc) return rc;
        rc=aclSetOutputTensorAddr(plan.exec,0,plan.out,out+2*t*64*64);
        if (rc) return rc;
        rc=aclnnEmbeddingDenseBackward(workspace,plan.bytes,plan.exec,stream);
        if (rc) return rc;
        ++b->launches;
    }
    return 0;
}
extern "C" int packet_close(Batch* b) {
    if (!b) return -1;
    const auto n=b->launches;
    for (auto& p:b->entries) p.release();
    delete b;
    return n;
}
