namespace dynamic_embedding_rebase {
namespace {
Counters copy_counters;
constexpr uint64_t copy_domain = 0x434f505931323842ULL;

bool residual_copy_view(const at::Tensor &t, View &view)
{
    if (t.layout() != at::kStrided || t.device().type() != c10::DeviceType::PrivateUse1 ||
        t.is_neg() || t.is_conj() || !t.is_contiguous() || t.dim() < 1 || t.dim() > 4 ||
        at_npu::native::FormatHelper::GetFormat(t) != ACL_FORMAT_ND || t.storage_offset() < 0) {
        return false;
    }
    for (auto stride : t.strides()) {
        if (stride <= 0) {
            return false;
        }
    }
    const auto item = t.itemsize();
    const auto capacity = t.storage().nbytes();
    const auto offset = static_cast<size_t>(t.storage_offset());
    const auto count = static_cast<size_t>(t.numel());
    if (!count || capacity % item || offset > capacity / item || count > capacity / item - offset) {
        return false;
    }
    const uintptr_t base = reinterpret_cast<uintptr_t>(t.storage().data());
    const size_t advance = offset * item;
    const size_t bytes = count * item;
    if (!base || base % 512 || bytes != 8192 ||
        advance > std::numeric_limits<uintptr_t>::max() - base ||
        bytes > std::numeric_limits<uintptr_t>::max() - (base + advance)) {
        return false;
    }
    // Preserve the original allocation alignment and the exact logical data address.
    // The residual prefix is outside the writable logical tensor. No trailing padding is added.
    const auto residual = advance % 512;
    view.tensor = t.unsafeGetTensorImpl();
    view.base = reinterpret_cast<void *>(base + advance - residual);
    view.offset = static_cast<int64_t>(residual / item);
    view.storage_elements = view.offset + t.numel();
    view.counters = &copy_counters;
    view.domain = copy_domain;
    return true;
}

bool eligible_copy(const at::Tensor &dst, const at::Tensor &src, View &to, View &from)
{
    const auto dtype = src.scalar_type();
    if (dtype != at::kBFloat16 || dtype != dst.scalar_type() || src.dim() != 1 || src.size(0) != 4096 ||
        src.layout() != at::kStrided || dst.layout() != at::kStrided ||
        dst.device() != src.device() || !dst.sizes().equals(src.sizes()) ||
        dst.storage().unsafeGetStorageImpl() == src.storage().unsafeGetStorageImpl() ||
        !producer_path() || !residual_copy_view(dst, to) || !residual_copy_view(src, from)) {
        return false;
    }
    // Handle only the two offset classes observed in the 1024-table native audit.
    if (to.offset != 64 || (from.offset != 0 && from.offset != 64)) {
        return false;
    }
    const auto size = src.nbytes();
    const auto x = reinterpret_cast<uintptr_t>(to.base) + to.offset * dst.itemsize();
    const auto y = reinterpret_cast<uintptr_t>(from.base) + from.offset * src.itemsize();
    if (x < y + size && y < x + size) {
        return false;
    }
    return true;
}
}  // namespace

Scope::Scope(ExactCopy, const at::Tensor &dst, const at::Tensor &src) : previous_(active)
{
    static const bool enabled = flag("NPU_INPLACE_COPY_REBASE");
    if (enabled) {
        if (eligible_copy(dst, src, views_[0], views_[1])) {
            count_ = 2;
            increment(copy_counters.eligible);
        } else {
            increment(copy_counters.fallback);
        }
    }
    active = this;
}

extern "C" __attribute__((visibility("default"))) uint64_t npu_inplace_copy_rebase_counter(unsigned index)
{
    return count(copy_counters, index);
}
}  // namespace dynamic_embedding_rebase
