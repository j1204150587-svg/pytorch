#include <ATen/ATen.h>
#include <ATen/TensorSubclassLikeUtils.h>
#include <ATen/record_function.h>
#include <c10/util/strides.h>
#include <torch/library.h>

namespace {
bool plain_flat(const at::Tensor &t)
{
    return t.defined() && t.device().is_privateuseone() && t.layout() == at::kStrided &&
           t.scalar_type() == at::kBFloat16 && !t.is_inference() && !t.has_names() &&
           !t.is_nested() && !t.is_conj() && !t.is_neg() && !at::isTensorSubclassLike(t) &&
           !t.requires_grad() && !(t.is_view() && t._base().requires_grad()) &&
           !t._fw_grad(0).defined() && t.dim() == 1 && t.stride(0) == 1 && t.storage_offset() >= 0;
}

bool plain_meta(const at::Tensor &t)
{
    // Bare Meta is itself in isTensorSubclassLike(). Remove only Meta's backend
    // bit, preserving its sparse / Python / functional / functorch exclusions.
    constexpr auto disallowed = at::kTensorSubclassLike.remove_backend(c10::BackendComponent::MetaBit);
    return t.defined() && t.device().is_meta() && t.layout() == at::kStrided &&
           t.scalar_type() == at::kBFloat16 && !t.is_inference() && !t.has_names() && !t.is_nested() &&
           !t.is_neg() && !t.is_conj() && !t.requires_grad() && !t._fw_grad(0).defined() &&
           !t.unsafeGetTensorImpl()->has_symbolic_sizes_strides() &&
           (t.unsafeGetTensorImpl()->key_set() & disallowed).empty();
}

std::vector<at::Tensor> try_unflatten(const at::Tensor &flat, at::TensorList shapes)
{
    if (shapes.size() < 32 || !plain_flat(flat)) {
        return {};
    }
    int64_t remaining = flat.numel();
    for (const auto &shape : shapes) {
        if (!plain_meta(shape)) {
            return {};
        }
        const auto n = shape.numel();
        if (n <= 0 || n > remaining) {
            return {};
        }
        remaining -= n;
    }
    // Every offset fits inside a backing storage known to contain this 1-D flat
    // tensor. Native unflatten permits unused trailing elements.
    RECORD_FUNCTION("NPUDynamicMetaUnflattenViews", std::vector<c10::IValue>());
    std::vector<at::Tensor> outputs;
    outputs.reserve(shapes.size());
    int64_t offset = flat.storage_offset();
    for (const auto &shape : shapes) {
        const auto dims = shape.sizes();
        outputs.emplace_back(flat.as_strided(dims, c10::contiguous_strides(dims), offset));
        offset += shape.numel();
    }
    return outputs;
}
}  // namespace

TORCH_LIBRARY(npu_dynamic_meta_views, m)
{
    m.def("try_unflatten(Tensor(a) flat, Tensor[] shapes) -> Tensor(a)[]");
}

TORCH_LIBRARY_IMPL(npu_dynamic_meta_views, CompositeImplicitAutograd, m)
{
    m.impl("try_unflatten", TORCH_FN(try_unflatten));
}
