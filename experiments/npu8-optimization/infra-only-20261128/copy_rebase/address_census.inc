#include <fstream>
#include <map>
#include <mutex>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace dynamic_embedding_rebase {
namespace {
struct AddressEvent {
    long tid;
    int64_t pta;
    int32_t physical;
    uint32_t dst_begin, dst_end, src_begin, src_end;
    size_t dst_bytes, src_bytes;
    bool same_impl, same_storage, same_metadata, contiguous, same_dtype;
    int dst_format, src_format;
};
std::mutex address_mutex;
std::map<uintptr_t,uint32_t> address_symbols;
std::vector<AddressEvent> address_rows;

uint32_t symbol(uintptr_t address)
{
    return address_symbols.emplace(address, static_cast<uint32_t>(address_symbols.size()+1)).first->second;
}

void write_addresses()
{
    const char *dir = std::getenv("INFRA_COUNTER_DIR");
    if (!dir) return;
    std::lock_guard<std::mutex> guard(address_mutex);
    std::ofstream out(std::string(dir)+"/copy-address-"+std::to_string(getpid())+".tsv");
    out << "ordinal\ttid\tpta_stream\tphysical_stream\tdst_start\tdst_end\tsrc_start\tsrc_end"
           "\tdst_bytes\tsrc_bytes\tsame_impl\tsame_storage\tsame_metadata\tcontiguous\tsame_dtype\tdst_format\tsrc_format\n";
    size_t order=0;
    for (const auto &r: address_rows) {
        out << order++ << '\t' << r.tid << '\t' << r.pta << '\t' << r.physical << '\t'
            << r.dst_begin << '\t' << r.dst_end << '\t' << r.src_begin << '\t' << r.src_end << '\t'
            << r.dst_bytes << '\t' << r.src_bytes << '\t' << r.same_impl << '\t' << r.same_storage << '\t'
            << r.same_metadata << '\t' << r.contiguous << '\t' << r.same_dtype << '\t'
            << r.dst_format << '\t' << r.src_format << '\n';
    }
}

void observe_addresses(const at::Tensor &dst, const at::Tensor &src)
{
    static const bool enabled=flag("NPU_INPLACE_COPY_ADDRESS");
    if (!enabled || dst.layout()!=at::kStrided || src.layout()!=at::kStrided ||
        dst.device().type()!=c10::DeviceType::PrivateUse1 || src.device().type()!=c10::DeviceType::PrivateUse1) {
        return;
    }
    static const int registered = std::atexit(write_addresses);
    (void)registered;
    const auto stream=c10_npu::getCurrentNPUStream();
    int32_t physical=-1;
    if (aclrtStreamGetId(stream.stream(false), &physical)!=ACL_ERROR_NONE) {
        physical=-1;
    }
    const uintptr_t dptr=reinterpret_cast<uintptr_t>(dst.data_ptr());
    const uintptr_t sptr=reinterpret_cast<uintptr_t>(src.data_ptr());
    const auto db=dst.nbytes(), sb=src.nbytes();
    const auto fd=at_npu::native::FormatHelper::GetFormat(dst), fs=at_npu::native::FormatHelper::GetFormat(src);
    const auto dtype=dst.scalar_type()==src.scalar_type();
    const auto same=dtype && fd==fs && dst.device()==src.device() && dst.sizes().equals(src.sizes()) &&
                    dst.strides().equals(src.strides()) && dst.is_neg()==src.is_neg() && dst.is_conj()==src.is_conj();
    std::lock_guard<std::mutex> guard(address_mutex);
    // Assign stable IDs to *real addresses*. The same address gets the same ID even if it
    // appears first as an exclusive end and later as a start. Raw pointers never leave the worker.
    address_rows.push_back({syscall(SYS_gettid),stream.id(),physical,symbol(dptr),symbol(dptr+db),symbol(sptr),
        symbol(sptr+sb),db,sb,dst.unsafeGetTensorImpl()==src.unsafeGetTensorImpl(),
        dst.storage().unsafeGetStorageImpl()==src.storage().unsafeGetStorageImpl(),same,
        dst.is_contiguous()&&src.is_contiguous(),dtype,static_cast<int>(fd),static_cast<int>(fs)});
}
}  // namespace
}  // namespace dynamic_embedding_rebase
