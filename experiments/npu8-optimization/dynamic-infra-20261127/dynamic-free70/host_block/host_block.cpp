#include <ATen/ATen.h>
#include <torch/library.h>

namespace {
using at::Tensor;

Tensor rms(const Tensor& x, const Tensor& w, double eps) {
  Tensor variance = x.pow(2).mean({-1}, true);
  return (x * (variance + eps).rsqrt()) * w;
}

Tensor block(const Tensor& x, const Tensor& in_norm, const Tensor& uvqk_w,
             const Tensor& uvqk_b, const Tensor& out_norm, const Tensor& out_w,
             const Tensor& out_b, double in_eps, double out_eps) {
  TORCH_CHECK(x.dim() == 3 && x.size(2) == 64, "unexpected input to the HSTU block");
  auto b = x.size(0), l = x.size(1);
  auto uvqk = at::linear(rms(x, in_norm, in_eps), uvqk_w, uvqk_b);
  auto parts = at::split_with_sizes(uvqk, {64,64,64,64}, -1);
  const auto& u = parts[0];
  auto q = parts[2].view({b,l,4,16}).transpose(1,2);
  auto k = parts[3].view({b,l,4,16}).transpose(1,2);
  auto v = parts[1].view({b,l,4,16}).transpose(1,2);
  auto scores = at::matmul(q,k.transpose(-2,-1)) * 0.25;
  auto mask = at::tril(at::ones({l,l},x.options().dtype(at::kBool)));
  auto activated = at::silu(scores);
  auto causal = activated.masked_fill(at::bitwise_not(mask),0.0);
  auto attn = at::matmul(causal/static_cast<double>(l),v).transpose(1,2).reshape({b,l,-1});
  auto normed = rms(attn,out_norm,out_eps);
  auto gate = at::silu(u);
  auto gated = normed * gate;
  auto y = at::cat({gated,u,x},-1);
  // The experiment guard requires standard training-mode dropout with p=0.
  return x + at::dropout(at::linear(y,out_w,out_b),0.0,true);
}
} // namespace

TORCH_LIBRARY(npu_hstu_native_block, m) {
  m.def("block(Tensor x, Tensor in_norm, Tensor uvqk_w, Tensor uvqk_b, Tensor out_norm, "
        "Tensor out_w, Tensor out_b, float in_eps, float out_eps) -> Tensor");
}

TORCH_LIBRARY_IMPL(npu_hstu_native_block, CompositeImplicitAutograd, m) {
  m.impl("block",TORCH_FN(block));
}
