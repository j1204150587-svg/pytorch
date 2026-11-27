#include "torch_npu/csrc/core/npu/NPUCachingAllocator.h"
#include "torch_npu/csrc/core/npu/NPUGraphsUtils.h"
#include "torch_npu/csrc/core/npu/sys_ctrl/npu_sys_ctrl.h"
#include "torch_npu/csrc/framework/FormatHelper.h"
#include <array>
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <mutex>

namespace repeated_embedding {
using Prepare = int (*)(const aclTensor *,const aclTensor *,int64_t,int64_t,bool,
                        aclTensor *,uint64_t *,aclOpExecutor **);
using Set = int (*)(aclOpExecutor *,size_t,aclTensor *,void *);
using Control = int (*)(aclOpExecutor *);
struct Api {
    _aclCreateTensor create=GET_OP_API_FUNC(aclCreateTensor);
    _aclDestroyTensor destroy_tensor=GET_OP_API_FUNC(aclDestroyTensor);
    Prepare prepare=reinterpret_cast<Prepare>(GetOpApiFuncAddr("aclnnEmbeddingDenseBackwardGetWorkspaceSize"));
    OpApiFunc run=reinterpret_cast<OpApiFunc>(GetOpApiFuncAddr("aclnnEmbeddingDenseBackward"));
    Control repeat=reinterpret_cast<Control>(GetOpApiFuncAddr("aclSetAclOpExecutorRepeatable"));
    Control destroy=reinterpret_cast<Control>(GetOpApiFuncAddr("aclDestroyAclOpExecutor"));
    Set input=reinterpret_cast<Set>(GetOpApiFuncAddr("aclSetInputTensorAddr"));
    Set output=reinterpret_cast<Set>(GetOpApiFuncAddr("aclSetOutputTensorAddr"));
    bool valid() const { return create && destroy_tensor && prepare && run && repeat && destroy && input && output; }
};

enum Count { eligible, fallback, prepared, reused, launches, drained, rejected, rebind_failure, live, peak_slots,
             peak_workspace, runtime_fallback, enumerated };
std::array<std::atomic<uint64_t>,enumerated> counts{};
extern "C" __attribute__((visibility("default"))) uint64_t npu_dynamic_repeat_counter(unsigned i)
{
    return i<enumerated ? counts[i].load(std::memory_order_relaxed) : 0;
}

void maximum(Count field,uint64_t value)
{
    auto old=counts[field].load(std::memory_order_relaxed);
    while (old<value && !counts[field].compare_exchange_weak(old,value,std::memory_order_relaxed)) {}
}

struct Input {
    at::Tensor grad,ids,out;  // Hold host queue arguments until their submission has completed.
    aclrtStream stream;
    int64_t cube,vector;
    int64_t index_offset() const { return ids.storage_offset(); }
    int64_t residual() const { return index_offset()%64; }
    void *address(unsigned slot,bool bounded) const {
        const auto &t=slot==0 ? grad : slot==1 ? ids : out;
        auto *p=static_cast<const char *>(t.storage().data());
        return const_cast<char *>(p+(slot==1 && bounded ? (index_offset()-residual())*8 : 0));
    }
};

struct Entry {
    aclOpExecutor *executor=nullptr;
    std::array<aclTensor *,3> descriptors{};
    at::Tensor workspace;
    uint64_t bytes=0;
    bool retained=false;
    void release(Api &api) {
        if (retained && executor) {
            NPU_CHECK_ERROR(api.destroy(executor),"aclDestroyAclOpExecutor");
            counts[live]--;
        }
        for (auto *desc:descriptors) if (desc) api.destroy_tensor(desc);
        *this=Entry{};
    }
};

struct Pool {
    Api api;
    std::mutex mutex;
    std::atomic<bool> disabled{false};
    std::array<Entry,16> slots{};
    std::array<int64_t,4> epoch{};
    aclrtStream stream=nullptr;
    aclrtContext context=nullptr;
    int device=-1;
    uint64_t workspace_bytes=0;
    void *shared_workspace=nullptr;
    bool initialized=false;
    bool active_epoch=false;

    void clear(bool release_workspace=false) {
        if (active_epoch) {
            // Raw ACL only: re-entering the PTA queue from its consumer would deadlock.
            NPU_CHECK_ERROR(aclrtSynchronizeStream(stream),"repeat epoch synchronize");
            counts[drained]++;
            for (auto &slot:slots) slot.release(api);
        }
        active_epoch=false;
        if (release_workspace && shared_workspace) {
            NPU_CHECK_ERROR(aclrtFree(shared_workspace),"release repeat stream workspace");
            shared_workspace=nullptr;
            workspace_bytes=0;
        }
    }
    void init(aclrtStream s,int d) {
        initialized=true; stream=s; device=d;
        NPU_CHECK_ERROR(aclrtGetCurrentContext(&context),"get repeat owner context");
        c10_npu::NpuSysCtrl::GetInstance().RegisterReleaseFn([this](){
            std::lock_guard<std::mutex> lock(mutex);
            disabled=true;
            aclrtContext prior=nullptr;
            if (aclrtGetCurrentContext(&prior)!=0) prior=nullptr;
            if (prior!=context) NPU_CHECK_ERROR(aclrtSetCurrentContext(context),"select repeat owner context");
            clear(true);
            if (prior && prior!=context) NPU_CHECK_ERROR(aclrtSetCurrentContext(prior),"restore context");
        },c10_npu::ReleasePriority::PriorityFirst);
    }
    int fresh(Entry &entry,const Input &x,bool bounded,bool retain,bool share_workspace) {
        for (unsigned j=0;j<3;j++) {
            const auto &t=j==0?x.grad:j==1?x.ids:x.out;
            int64_t offset=t.storage_offset();
            int64_t extent=t.storage().nbytes()/t.itemsize();
            if (j==1 && bounded) {
                offset=x.residual();
                const auto need=offset+4+(t.size(0)*t.size(1)-1)*4096;
                extent=((need+63)/64)*64;
                TORCH_CHECK(extent <= static_cast<int64_t>(t.storage().nbytes()/8)-(t.storage_offset()-offset),
                            "invalid repeatable index storage extent");
            }
            const auto format=j==0?ACL_FORMAT_NCHW:j==1?ACL_FORMAT_NCL:ACL_FORMAT_ND;
            entry.descriptors[j]=api.create(t.sizes().data(),t.dim(),j==1?ACL_INT64:ACL_BF16,
                t.strides().data(),offset,format,&extent,1,x.address(j,bounded));
            TORCH_CHECK(entry.descriptors[j],"could not construct repeatable embedding descriptor");
        }
        auto &a=entry.descriptors;
        int result=api.prepare(a[0],a[1],64,-1,false,a[2],&entry.bytes,&entry.executor);
        NPU_CHECK_ERROR(result,"embedding repeat preparation");
        counts[prepared]++;
        if (retain) {
            result=api.repeat(entry.executor);
            if (!result) {
                entry.retained=true;
                counts[live]++;
                maximum(peak_slots,counts[live].load());
            } else {
                counts[rejected]++;
                disabled=true;
            }
        }
        if (share_workspace && entry.bytes>workspace_bytes) {
            // All entries submit on this one stream. Retain the maximum demand,
            // never reuse the PTA allocator's resizable per-stream raw pointer.
            if (shared_workspace) {
                NPU_CHECK_ERROR(aclrtSynchronizeStream(x.stream),"resize repeat workspace");
                NPU_CHECK_ERROR(aclrtFree(shared_workspace),"release smaller repeat workspace");
                shared_workspace=nullptr;
                workspace_bytes=0;
            }
            const int alloc_result=aclrtMalloc(&shared_workspace,entry.bytes,ACL_MEM_MALLOC_NORMAL_ONLY);
            if (alloc_result) {
                entry.release(api);
                NPU_CHECK_ERROR(alloc_result,"allocate repeat-owned workspace");
            }
            workspace_bytes=entry.bytes;
            maximum(peak_workspace,workspace_bytes);
        } else if (entry.bytes && !share_workspace) {
            entry.workspace=at_npu::native::OpPreparation::unsafe_empty_workspace(entry.bytes,x.stream);
        }
        return 0;
    }
    int run(Entry &entry,const Input &x,bool share_workspace=false) {
        void *ws=share_workspace?shared_workspace:
            (entry.workspace.defined()?const_cast<void *>(entry.workspace.storage().data()):nullptr);
        const int result=api.run(ws,entry.bytes,entry.executor,x.stream);
        // No retry after stage 2; an error may follow a partial submission.
        NPU_CHECK_ERROR(result,"repeatable embedding stage 2");
        counts[launches]++;
        if (!entry.retained) {
            entry.executor=nullptr;  // Single-use executor was consumed by stage 2.
            entry.release(api);
        }
        return result;
    }
    int once(const Input &x) {
        counts[runtime_fallback]++;
        Entry e;
        fresh(e,x,false,false,false);
        return run(e,x);
    }
    int consume(const Input &x,bool enable_repeat) {
        std::lock_guard<std::mutex> lock(mutex);
        if (c10_npu::check_dequeue_need_use(x.stream)) c10_npu::UseStreamResInCurrentThread(x.stream);
        if (disabled.load()) return once(x);
        if (!initialized) init(x.stream,x.ids.get_device());
        if (x.stream!=stream || x.ids.get_device()!=device) return once(x);
        std::array<int64_t,4> key{x.ids.size(0),x.ids.size(1),x.cube,x.vector};
        if (epoch!=key) { clear(); epoch=key; }
        active_epoch=true;
        auto &entry=slots[x.residual()/4];
        if (entry.retained) {
            bool ok=true;
            for (unsigned j=0;j<3 && ok;j++) {
                const auto set=j==2?api.output:api.input;
                ok=set(entry.executor,j==2?0:j,entry.descriptors[j],x.address(j,true))==0;
            }
            if (!ok) {
                counts[rebind_failure]++;
                disabled=true;
                clear();
                return once(x);
            }
            counts[reused]++;
        } else {
            fresh(entry,x,true,enable_repeat,true);
            if (enable_repeat && !entry.retained) counts[runtime_fallback]++;
        }
        return run(entry,x,true);
    }
};

Pool &pool() {
    // RegisterReleaseFn releases all device/CANN owners before stream destruction.
    // The process-lifetime shell contains no live executor after that callback.
    static Pool *p=new Pool;
    return *p;
}

bool plain(const at::Tensor &t) {
    return t.defined() && t.device().is_privateuseone() && t.layout()==at::kStrided &&
        !t.is_conj() && !t.is_neg() && t.storage().data() &&
        reinterpret_cast<uintptr_t>(t.storage().data())%512==0 &&
        t.storage().data_ptr().get_deleter()==c10_npu::NPUCachingAllocator::get()->raw_deleter() &&
        at_npu::native::FormatHelper::IsOpInputBaseFormat(t);
}

bool submit(const at::Tensor &g,const at::Tensor &i,int64_t rows,int64_t padding,bool scale,const at::Tensor &out) {
    static const int mode=[](){
        const auto *v=std::getenv("NPU_DYNAMIC_REPEAT_EMBEDDING");
        if (v && (std::strcmp(v,"1")==0 || std::strcmp(v,"repeat")==0)) return 2;
        return v && std::strcmp(v,"fresh")==0 ? 1 : 0;
    }();
    if (!mode) return false;
    if (rows!=64 || padding!=-1 || scale || !plain(i) || !plain(g) || !plain(out) ||
        i.device()!=g.device() || g.device()!=out.device() || i.dim()!=3 || g.dim()!=4 || out.dim()!=2 ||
        i.scalar_type()!=at::kLong || g.scalar_type()!=at::kBFloat16 || out.scalar_type()!=at::kBFloat16 ||
        i.size(0)<1 || i.size(0)>4 || i.size(1)<12 || i.size(1)>32 || i.size(2)!=4 ||
        i.stride(0)!=i.size(1)*4096 || i.stride(1)!=4096 || i.stride(2)!=1 ||
        i.storage_offset()<0 || i.storage_offset()>=4096 || i.storage_offset()%4 ||
        i.storage().nbytes()!=static_cast<size_t>(i.size(0)*i.size(1)*4096*8) ||
        g.size(0)!=i.size(0) || g.size(1)!=i.size(1) || g.size(2)!=4 || g.size(3)!=64 ||
        g.stride(0)!=i.size(1)*256 || g.stride(1)!=256 || g.stride(2)!=64 || g.stride(3)!=1 ||
        g.storage_offset()!=0 || g.storage().nbytes()!=static_cast<size_t>(g.numel()*2) ||
        out.size(0)!=64 || out.size(1)!=64 || out.stride(0)!=64 || out.stride(1)!=1 ||
        out.storage_offset()!=0 || out.storage().nbytes()!=8192 ||
        at::globalContext().deterministicAlgorithms() || c10_npu::option::OptionsManager::GetTaskQueueEnable()!=1 ||
        c10_npu::currentStreamCaptureStatus()!=c10_npu::CaptureStatus::None || !pool().api.valid() || pool().disabled) {
        counts[fallback]++;
        return false;
    }
    auto stream=c10_npu::getCurrentNPUStream();
    if (stream.device_index()!=i.get_device()) { counts[fallback]++; return false; }
    auto raw=stream.stream(false);
    if (c10_npu::check_enqueue_need_use(raw)) c10_npu::UseStreamResInCurrentThread(raw);
    int64_t cube=-1,vector=-1;
    if (c10_npu::is_core_control_enabled()) {
        cube=c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_CUBE_CORE);
        vector=c10_npu::GetResInCurrentThread(c10_npu::acl::ACL_RT_DEV_RES_VECTOR_CORE);
    }
    at_npu::native::SetDeterministic();
    // Keep allocator blocks live on the actual submission stream, including cross-allocation-stream inputs.
    for (const auto *t:{&g,&i,&out}) c10_npu::NPUCachingAllocator::recordStream(t->storage().data_ptr(),stream);
    Input args{g,i,out,raw,cube,vector};
    counts[eligible]++;
    at_npu::native::OpCommand::RunOpApiV2("aclnnEmbeddingDenseBackward",[args](){ return pool().consume(args,mode==2); });
    return true;
}
} // namespace repeated_embedding
