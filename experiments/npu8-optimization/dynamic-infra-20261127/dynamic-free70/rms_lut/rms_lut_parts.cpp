#include <ATen/ATen.h>
#include <torch/library.h>
#include <tuple>

namespace {
using at::Tensor;
using Three=std::tuple<Tensor,Tensor,Tensor>;

Three forward_parts(const Tensor& x, const Tensor& w, double eps) {
  auto variance=x.pow(2).mean({-1},true);
  auto inv=(variance+eps).rsqrt();
  auto z=x*inv;
  return {z*w,inv,z};
}

Three backward_parts(const Tensor& x, const Tensor& w, const Tensor& inv,
                     const Tensor& z, const Tensor& dy) {
  // These are the v2 BF16 operations, with independent direct and variance
  // edges for the HSTU residual gradient that may already be present.
  auto gz=dy*w;
  auto dw=(dy*z).sum_to_size(w.sizes());
  auto direct=gz*inv;
  auto dr=(gz*x).sum_to_size(inv.sizes());
  auto dvar=(dr*-0.5)*inv.pow(3);
  auto dsq=dvar/x.size(-1);
  return {direct,dsq*(x*2),dw};
}
}

TORCH_LIBRARY(npu_hstu_rms_lut,m) {
  m.def("forward_parts(Tensor x, Tensor w, float eps) -> (Tensor, Tensor, Tensor)",TORCH_FN(forward_parts));
  m.def("backward_parts(Tensor x, Tensor w, Tensor inv, Tensor z, Tensor dy) -> (Tensor, Tensor, Tensor)",TORCH_FN(backward_parts));
}
