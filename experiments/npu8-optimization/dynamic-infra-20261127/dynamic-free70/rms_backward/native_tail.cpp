#include <ATen/ATen.h>
#include <torch/library.h>

namespace {
using at::Tensor;
std::tuple<Tensor, Tensor, Tensor> finish(const Tensor& x, const Tensor& w, const Tensor& inv,
                                        const Tensor& direct, const Tensor& row, const Tensor& weight) {
  auto dw = weight.sum_to_size(w.sizes());
  auto dr = row.sum_to_size(inv.sizes());
  auto dvar = (dr * -0.5) * inv.pow(3);
  auto dsq = dvar / x.size(-1);
  return {direct, dsq * (x * 2), dw};
}
}

TORCH_LIBRARY(npu_hstu_rms_grad_prefix, m) {
  m.def("finish(Tensor x, Tensor weight, Tensor inv, Tensor direct, Tensor row, Tensor weight_products)"
        " -> (Tensor, Tensor, Tensor)", TORCH_FN(finish));
}
