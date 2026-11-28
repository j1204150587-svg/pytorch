#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>
#include <sys/syscall.h>
#include <unistd.h>
#include "torch_npu/csrc/core/npu/NPUStream.h"

namespace dynamic_embedding_rebase {
namespace {
struct MetaClass { size_t index; uint64_t calls; };
struct CopyEvent { long thread; int64_t stream; size_t index; };
std::map<std::string, MetaClass> copy_census;
std::vector<CopyEvent> copy_events;
std::mutex copy_census_mutex;

void write_census()
{
    const char *dir = std::getenv("INFRA_COUNTER_DIR");
    if (dir == nullptr) {
        return;
    }
    std::lock_guard<std::mutex> guard(copy_census_mutex);
    std::ofstream out(std::string(dir) + "/copy-guard-" + std::to_string(getpid()) + ".tsv");
    out << "class_id\tcount\tsame_storage\tdst\tsrc\n";
    for (const auto &entry : copy_census) {
        out << entry.second.index << '\t' << entry.second.calls << '\t' << entry.first << '\n';
    }
    std::ofstream events(std::string(dir) + "/copy-events-" + std::to_string(getpid()) + ".tsv");
    events << "ordinal\thost_tid\tpta_stream_id\tclass_id\n";
    size_t ordinal = 0;
    for (const auto &event : copy_events) {
        events << ordinal++ << '\t' << event.thread << '\t' << event.stream << '\t' << event.index << '\n';
    }
}

std::string metadata(const at::Tensor &t)
{
    std::ostringstream key;
    key << c10::toString(t.scalar_type()) << ":";
    for (int64_t size : t.sizes()) {
        key << size << ",";
    }
    if (t.layout() != at::kStrided || t.device().type() != c10::DeviceType::PrivateUse1) {
        return key.str() + ":unsupported";
    }
    const auto item = t.itemsize();
    const auto offset = t.storage_offset();
    const auto bytes = t.nbytes();
    key << ":ct=" << t.is_contiguous() << ":fmt=" << at_npu::native::FormatHelper::GetFormat(t)
        << ":store=" << t.storage().nbytes() << ":bytes=" << bytes
        << ":base512=" << reinterpret_cast<uintptr_t>(t.storage().data()) % 512
        << ":off512=" << ((offset % 512) * item) % 512
        << ":off0=" << (offset == 0) << ":ptr512=" << reinterpret_cast<uintptr_t>(t.data_ptr()) % 512
        << ":len512=" << bytes % 512 << ":neg=" << t.is_neg() << ":conj=" << t.is_conj();
    return key.str();
}

void diagnose_copy(const at::Tensor &dst, const at::Tensor &src)
{
    static const bool enabled = flag("NPU_INPLACE_COPY_DIAG");
    if (!enabled) {
        return;
    }
    static const int registered = std::atexit(write_census);
    (void)registered;
    const std::string key = std::to_string(dst.storage().unsafeGetStorageImpl() == src.storage().unsafeGetStorageImpl()) +
                            "\t" + metadata(dst) + "\t" + metadata(src);
    const long thread = syscall(SYS_gettid);
    const auto stream = c10_npu::getCurrentNPUStream().id();
    std::lock_guard<std::mutex> guard(copy_census_mutex);
    auto item = copy_census.emplace(key, MetaClass{copy_census.size(),0}).first;
    ++item->second.calls;
    copy_events.push_back({thread, stream, item->second.index});
}
}  // namespace
}  // namespace dynamic_embedding_rebase
