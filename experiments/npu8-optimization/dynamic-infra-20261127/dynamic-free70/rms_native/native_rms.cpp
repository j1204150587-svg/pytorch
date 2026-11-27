#include <ATen/ATen.h>
#include <torch/csrc/autograd/custom_function.h>
#include <torch/library.h>

namespace {
using at::Tensor;
using torch::autograd::AutogradContext;
using torch::autograd::variable_list;

class NativeRowRms : public torch::autograd::Function<NativeRowRms> {
 public:
  static Tensor forward(AutogradContext* ctx, Tensor x_direct, Tensor x_variance, Tensor w, double eps) {
    // All six operations use the stock dispatched ATen implementations.
    auto square = x_variance.pow(2);
    auto mean = square.mean({-1}, true);
    auto inv = (mean + eps).rsqrt();
    auto z = x_direct * inv;
    ctx->save_for_backward({x_variance, w, inv, z});
    return z * w;
  }

  static variable_list backward(AutogradContext* ctx, variable_list outputs) {
    auto saved = ctx->get_saved_variables();
    const auto& x = saved[0];
    const auto& w = saved[1];
    const auto& inv = saved[2];
    const auto& z = saved[3];
    const auto& dy = outputs[0];
    auto gz = dy * w;
    auto dw = (dy * z).sum_to_size(w.sizes());
    auto direct = gz * inv;
    auto dr = (gz * x).sum_to_size(inv.sizes());
    auto dvar = (dr * -0.5) * inv.pow(3);
    // Stock mean_backward expands [B,L,1] to width 64 before /64. Dividing
    // each row first keeps the BF16 boundary and lets mul broadcast it once.
    auto dsq = dvar / x.size(-1);
    auto dvar_x = dsq * (x * 2);
    // x also contributes to the HSTU residual. Keep the two original incoming
    // gradients separate: joining them here changes the BF16 association with
    // a residual gradient that may already be in the autograd input buffer.
    return {direct, dvar_x, dw, Tensor()};
  }
};

Tensor native_rms(const Tensor& x, const Tensor& w, double eps) {
  TORCH_CHECK(x.dim()==3 && x.size(-1)==64 && w.dim()==1 && w.size(0)==64 &&
              x.scalar_type()==at::kBFloat16 && w.scalar_type()==at::kBFloat16 &&
              x.device()==w.device(), "unexpected input to experimental RMS operator");
  return NativeRowRms::apply(x,x,w,eps);
}
} // namespace

TORCH_LIBRARY(npu_hstu_native_rms, m) {
  m.def("rms(Tensor x, Tensor weight, float eps) -> Tensor", TORCH_FN(native_rms));
}
