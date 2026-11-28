#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_add.h>
#include <array>
#include <chrono>
#include <cstdint>
#include <new>

struct PendingProbePlan {
    std::array<aclTensor*,3> values {};
    aclScalar* alpha = nullptr;
    aclOpExecutor* executor = nullptr;
    uint64_t ws_size = 0;
    void* workspace = nullptr;
};
extern "C" int pending_probe_prepare(int64_t b, int64_t l, void* x, void* y, void* out, PendingProbePlan** handle)
{
    auto* p = new (std::nothrow) PendingProbePlan;
    if (!p) return -1;
    *handle = p;
    const int64_t dims[] = {b,l,64}, strides[] = {l*64,64,1}, size=b*l*64, one=1;
    void* ptr[] = {x,y,out};
    for (unsigned j=0;j<3;++j) {
        p->values[j] = aclCreateTensor(dims,3,ACL_BF16,strides,0,ACL_FORMAT_NCL,&size,1,ptr[j]);
        if (!p->values[j]) return -2;
    }
    p->alpha = aclCreateScalar(const_cast<int64_t*>(&one),ACL_INT64);
    if (!p->alpha) return -3;
    int rc=aclnnAddGetWorkspaceSize(p->values[0],p->values[1],p->alpha,p->values[2],&p->ws_size,&p->executor);
    if (rc) return rc;
    rc=aclSetAclOpExecutorRepeatable(p->executor);
    if (rc) return rc;
    if (p->ws_size) return aclrtMalloc(&p->workspace,p->ws_size,ACL_MEM_MALLOC_NORMAL_ONLY);
    return 0;
}
static uint64_t nano()
{
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
extern "C" int pending_probe_window(PendingProbePlan** plans, unsigned size, unsigned count, aclrtStream stream,
                                     uint64_t* per_call, uint64_t* host_total, uint64_t* complete_total)
{
    if (!size || !count) return -10;
    int rc=aclrtSynchronizeStream(stream);
    if (rc) return rc;
    const auto begin=nano();
    for (unsigned i=0;i<count;++i) {
        auto* p=plans[i%size];
        const auto t=per_call?nano():0;
        rc=aclnnAdd(p->workspace,p->ws_size,p->executor,stream);
        if (per_call) per_call[i]=nano()-t;
        if (rc) return rc;
    }
    *host_total=nano()-begin;
    rc=aclrtSynchronizeStream(stream);
    *complete_total=nano()-begin;
    return rc;
}
extern "C" int pending_probe_close(PendingProbePlan* p)
{
    int rc=aclDestroyAclOpExecutor(p->executor);
    if (rc) return rc;
    if (p->workspace && (rc=aclrtFree(p->workspace))) return rc;
    for (auto* t:p->values) if ((rc=aclDestroyTensor(t))) return rc;
    if ((rc=aclDestroyScalar(p->alpha))) return rc;
    delete p;
    return 0;
}
