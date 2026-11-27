// Target headers for acl_meta and aclnnEmbeddingDenseBackward are force-included by build.py.
#include <array>
#include <cstdint>
#include <new>

struct Probe {
    std::array<aclTensor *,3> tensors{};
    aclOpExecutor *executor = nullptr;
    uint64_t bytes = 0;
    bool repeatable = false;
};

extern "C" void *probe_new() { return new (std::nothrow) Probe(); }

extern "C" int probe_tensor(Probe *p, int slot, const int64_t *dims, int rank,
                            const int64_t *strides, int64_t offset, int64_t extent, void *base)
{
    if (!p || slot<0 || slot>2 || p->tensors[slot] || !base || extent<=0) return -1;
    const auto format = rank==4 ? ACL_FORMAT_NCHW : rank==3 ? ACL_FORMAT_NCL : ACL_FORMAT_ND;
    const auto dtype = slot==1 ? ACL_INT64 : ACL_BF16;
    p->tensors[slot] = aclCreateTensor(dims,rank,dtype,strides,offset,format,&extent,1,base);
    return p->tensors[slot] ? 0 : -2;
}

extern "C" int probe_prepare(Probe *p, uint64_t *bytes)
{
    if (!p || !bytes || p->executor) return -1;
    const auto status = aclnnEmbeddingDenseBackwardGetWorkspaceSize(
        p->tensors[0],p->tensors[1],64,-1,false,p->tensors[2],&p->bytes,&p->executor);
    *bytes = p->bytes;
    return status;
}

extern "C" int probe_repeat(Probe *p)
{
    if (!p || !p->executor || p->repeatable) return -1;
    const auto status=aclSetAclOpExecutorRepeatable(p->executor);
    p->repeatable=status==0;
    return status;
}

extern "C" int probe_rebind(Probe *p,int slot,void *base)
{
    if (!p || !p->repeatable || slot<0 || slot>2 || !base || reinterpret_cast<uintptr_t>(base)%32) return -1;
    return slot==2 ? aclSetOutputTensorAddr(p->executor,0,p->tensors[2],base) :
                     aclSetInputTensorAddr(p->executor,slot,p->tensors[slot],base);
}

extern "C" int probe_launch(Probe *p,void *workspace,void *stream)
{
    if (!p || !p->executor || (p->bytes && !workspace)) return -1;
    const auto status=aclnnEmbeddingDenseBackward(workspace,p->bytes,p->executor,stream);
    if (!p->repeatable && status==0) p->executor=nullptr; // Normal stage 2 owns its executor.
    return status;
}

extern "C" int probe_close(Probe *p)
{
    if (!p) return -1;
    if (p->executor && !p->repeatable) return -3; // No safe public destructor for an unlaunched single-use handle.
    int status = p->executor ? aclDestroyAclOpExecutor(p->executor) : 0;
    if (status) return status;
    for (auto *desc:p->tensors) if (desc) aclDestroyTensor(desc);
    delete p;
    return 0;
}
