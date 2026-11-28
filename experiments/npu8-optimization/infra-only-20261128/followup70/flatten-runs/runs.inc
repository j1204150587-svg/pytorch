namespace {
bool flatFlag(const char* key)
{
    const char* value = std::getenv(key);
    return value && std::strcmp(value, "1") == 0;
}

struct FlatRun {
    size_t start;
    size_t end;
    int64_t low;
    int64_t elements;
    bool reverse;
};

at::Tensor tryRunFlatten(at::TensorList inputs)
{
    static const bool enabled = flatFlag("TORCH_NPU_FLATTEN_STORAGE_RUNS");
    static const bool audit = flatFlag("TORCH_NPU_FLATTEN_RUNS_AUDIT");
    // Keep the single-input view return and non-batched paths intact.
    if ((!enabled && !audit) || inputs.size() < 16) {
        return {};
    }
    const auto& first = inputs.front();
    if (!first.defined() || at::isTensorSubclassLike(first) || !first.has_storage() || first.device().type() != c10::DeviceType::PrivateUse1 ||
        first.scalar_type() != at::kBFloat16 || first.layout() != at::kStrided) {
        return {};
    }
    const auto storage = first.storage();
    const auto& desc = torch_npu::NPUBridge::GetNpuStorageImpl(first)->npu_desc_;
    if (desc.npu_format_ != ACL_FORMAT_ND || desc.origin_format_ != ACL_FORMAT_ND) {
        return {};
    }
    const auto capacity = storage.nbytes() / first.element_size();
    for (const auto& t : inputs) {
        if (!t.defined() || at::isTensorSubclassLike(t) || !t.has_storage() || t.layout() != at::kStrided || t.device() != first.device() ||
            t.scalar_type() != at::kBFloat16 || t.storage().unsafeGetStorageImpl() != storage.unsafeGetStorageImpl() ||
            t.requires_grad() || t._fw_grad(0).defined() || t.is_conj() || t.is_neg() || t.has_names() ||
            !t.is_contiguous() ||
            t.numel() <= 0 || t.storage_offset() < 0 ||
            static_cast<uint64_t>(t.storage_offset()) > capacity ||
            static_cast<uint64_t>(t.numel()) > capacity - static_cast<uint64_t>(t.storage_offset())) {
            return {};
        }
    }

    std::vector<FlatRun> runs;
    size_t ascending = 0, descending = 0;
    for (size_t i = 0; i < inputs.size();) {
        const int64_t width = inputs[i].numel();
        const int64_t offset = inputs[i].storage_offset();
        int64_t high = offset + width;
        int64_t low = offset;
        size_t j = i + 1;
        while (j < inputs.size() && inputs[j].storage_offset() == high) {
            high += inputs[j++].numel();
        }
        bool reverse = false;
        if (j == i + 1) {
            while (j < inputs.size() && inputs[j].numel() == width && low >= width &&
                   inputs[j].storage_offset() == low - width) {
                low -= width;
                ++j;
            }
            if (j - i >= 16) {
                reverse = true;
                descending += j - i;
            } else {
                low = offset;
                j = i + 1;
            }
        } else {
            ascending += j - i;
        }
        runs.push_back({i, j, low, high - low, reverse});
        i = j;
    }
    const bool useful = inputs.size() - runs.size() >= 15;
    if (audit) {
        std::fprintf(stderr, "PTA_FLATTEN_RUNS enabled=%d inputs=%zu runs=%zu ascending=%zu descending=%zu active=%d\n",
                     enabled, inputs.size(), runs.size(), ascending, descending, useful && enabled);
    }
    if (!enabled || !useful) {
        return {};
    }
    std::vector<at::Tensor> pieces;
    pieces.reserve(runs.size());
    for (const auto& run : runs) {
        if (run.reverse) {
            // Reverse rows while preserving order inside each original slice.
            const auto width = inputs[run.start].numel();
            auto matrix = first.as_strided({static_cast<int64_t>(run.end-run.start), width},
                                           {width, 1}, run.low);
            pieces.push_back(at::flip(matrix, {0}).view({-1}));
        } else {
            pieces.push_back(first.as_strided({run.elements}, {1}, run.low));
        }
    }
    // cat also allocates independent output when only one span remains.
    return at::cat(pieces);
}
} // namespace
