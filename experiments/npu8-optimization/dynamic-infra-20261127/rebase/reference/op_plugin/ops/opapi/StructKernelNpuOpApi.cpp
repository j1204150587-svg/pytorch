// Copyright (c) 2024 Huawei Technologies Co., Ltd
// All rights reserved.
//
// Licensed under the BSD 3-Clause License  (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <ATen/native/TypeProperties.h>
#include "op_plugin/AclOpsInterface.h"
#include "op_plugin/OpApiInterface.h"
#include "op_plugin/utils/op_api_common.h"

namespace op_api {
using npu_preparation = at_npu::native::OpPreparation;
using namespace op_plugin::utils;
using namespace op_infer;

at::Tensor _adaptive_avg_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnAdaptiveAvgPool2dBackward, acl_op::_adaptive_avg_pool2d_backward(grad_output, self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAdaptiveAvgPool2dBackward, grad_output, self, out);
    return out;
}

at::Tensor _adaptive_avg_pool3d(const at::Tensor & self, at::IntArrayRef output_size)
{
    DO_COMPATIBILITY(aclnnAdaptiveAvgPool3d, acl_op::_adaptive_avg_pool3d(self, output_size));
    auto output_size_0 = adaptive_avg_pool3d_npu_output_size(self, output_size);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAdaptiveAvgPool3d, self, output_size, out);
    return out;
}

at::Tensor _adaptive_avg_pool3d_backward(const at::Tensor & grad_output, const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnAdaptiveAvgPool3dBackward, acl_op::_adaptive_avg_pool3d_backward(grad_output, self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAdaptiveAvgPool3dBackward, grad_output, self, out);
    return out;
}

at::Tensor _add_relu(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha)
{
    DO_COMPATIBILITY(aclnnAddRelu, acl_op::_add_relu(self, other, alpha));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAddRelu, self, other, alpha, out);
    return out;
}

at::Tensor & _add_relu_out(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAddRelu, acl_op::_add_relu_out(self, other, alpha, out));
    auto output_size_0 = out.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, other}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAddRelu, self, other, alpha, out);
    return out;
}

at::Tensor & _add_relu_(at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha)
{
    DO_COMPATIBILITY(aclnnInplaceAddRelu, acl_op::_add_relu_(self, other, alpha));
    EXEC_NPU_CMD(aclnnInplaceAddRelu, self, other, alpha);
    return self;
}

::std::tuple<at::Tensor,at::Tensor> _aminmax(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnAminmaxAll, acl_op::_aminmax(self));
    auto output_size_0 = reduce_ops_npu_output_size(self, get_dimlist_for_tensor(self), false);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAminmaxAll, self, out0, out1);
    return std::make_tuple(std::move(out0), std::move(out1));
}

::std::tuple<at::Tensor,at::Tensor> _aminmax(const at::Tensor & self, int64_t dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnAminmaxDim, acl_op::_aminmax(self, dim, keepdim));
    auto output_size_0 = reduce_ops_npu_output_size(self, {dim}, keepdim);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAminmaxDim, self, dim, keepdim, out0, out1);
    return std::make_tuple(std::move(out0), std::move(out1));
}

at::Tensor _cdist_forward(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode)
{
    DO_COMPATIBILITY(aclnnCdist, acl_op::_cdist_forward(x1, x2, p, compute_mode));
    auto p_float = static_cast<float>(p);
    auto mode = std::abs(p - 2.0) < 1e-6 ? 2 : compute_mode.value_or(0);
    auto output_size_0 = op_infer::cdist_npu_output_size(x1, x2);
    auto output_dtype_0 = x1.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x1.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCdist, x1, x2, p_float, mode, out);
    return out;
}

at::Tensor _ctc_loss_backward(const at::Tensor & grad, const at::Tensor & log_probs, const at::Tensor & targets, at::IntArrayRef input_lengths, at::IntArrayRef target_lengths, const at::Tensor & neg_log_likelihood, const at::Tensor & log_alpha, int64_t blank, bool zero_infinity)
{
    DO_COMPATIBILITY(aclnnCtcLossBackward, acl_op::_ctc_loss_backward(grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity));
    auto output_size_0 = log_probs.sizes();
    auto output_dtype_0 = grad.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCtcLossBackward, grad, log_probs, targets, input_lengths, target_lengths, neg_log_likelihood, log_alpha, blank, zero_infinity, out);
    return out;
}

::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> _linalg_svd_out(const at::Tensor & A, bool full_matrices, bool compute_uv, c10::optional<c10::string_view> driver, at::Tensor & U, at::Tensor & S, at::Tensor & Vh)
{
    DO_COMPATIBILITY(aclnnSvd, acl_op::_linalg_svd_out(A, full_matrices, compute_uv, driver, U, S, Vh));
    auto V = Vh.dim() <= 1 ? Vh : Vh.transpose(-1, -2);
    auto output_size_0 = U.sizes();
    auto output_size_1 = S.sizes();
    auto output_size_2 = Vh.sizes();
    auto output_dtype_0 = U.scalar_type();
    auto output_dtype_1 = S.scalar_type();
    auto output_dtype_2 = Vh.scalar_type();
    npu_preparation::check_tensor({A}, U, output_dtype_0, output_size_0);
    npu_preparation::check_tensor({A}, S, output_dtype_1, output_size_1);
    npu_preparation::check_tensor({A}, Vh, output_dtype_2, output_size_2);
    EXEC_NPU_CMD(aclnnSvd, A, full_matrices, compute_uv, S, U, V);
    return std::forward_as_tuple(U, S, Vh);
}

at::Tensor _log_softmax(const at::Tensor & self, int64_t dim, bool half_to_float)
{
    DO_COMPATIBILITY(aclnnLogSoftmax, acl_op::_log_softmax(self, dim, half_to_float));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = half_to_float ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLogSoftmax, self, dim, out);
    return out;
}

at::Tensor & _log_softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLogSoftmax, acl_op::_log_softmax_out(self, dim, half_to_float, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = half_to_float ? at::kFloat : self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLogSoftmax, self, dim, out);
    return out;
}

at::Tensor _log_softmax_backward_data(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype)
{
    DO_COMPATIBILITY(aclnnLogSoftmaxBackward, acl_op::_log_softmax_backward_data(grad_output, output, dim, input_dtype));
    auto output_size_0 = grad_output.sizes();
    auto output_dtype_0 = grad_output.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLogSoftmaxBackward, grad_output, output, dim, out);
    return out;
}

at::Tensor & _log_softmax_backward_data_out(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLogSoftmaxBackward, acl_op::_log_softmax_backward_data_out(grad_output, output, dim, input_dtype, out));
    auto output_size_0 = grad_output.sizes();
    auto output_dtype_0 = grad_output.scalar_type();
    npu_preparation::check_tensor({grad_output, output}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLogSoftmaxBackward, grad_output, output, dim, out);
    return out;
}

at::Tensor _softmax(const at::Tensor & self, int64_t dim, bool half_to_float)
{
    DO_COMPATIBILITY(aclnnSoftmax, acl_op::_softmax(self, dim, half_to_float));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = half_to_float ? at::ScalarType::Float : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSoftmax, self, dim, out);
    return out;
}

at::Tensor & _softmax_out(const at::Tensor & self, int64_t dim, bool half_to_float, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSoftmax, acl_op::_softmax_out(self, dim, half_to_float, out));
    auto output_size_0 = out.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSoftmax, self, dim, out);
    return out;
}

at::Tensor & _softmax_backward_data_out(const at::Tensor & grad_output, const at::Tensor & output, int64_t dim, at::ScalarType input_dtype, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnSoftmaxBackward, acl_op::_softmax_backward_data_out(grad_output, output, dim, input_dtype, grad_input));
    EXEC_NPU_CMD(aclnnSoftmaxBackward, grad_output, output, dim, grad_input);
    return grad_input;
}

at::Tensor _prelu_kernel(const at::Tensor & self, const at::Tensor & weight)
{
    DO_COMPATIBILITY(aclnnPrelu, acl_op::_prelu_kernel(self, weight));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnPrelu, self, weight, out);
    return out;
}

at::Tensor _masked_scale(const at::Tensor & self, const at::Tensor & mask, double scale)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMaskedScale, self, mask, scale, out);
    return out;
}

at::Tensor abs(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = abs_out_dtype(self);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAbs, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & abs_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAbs, acl_op::abs_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = abs_out_dtype(self);
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAbs, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & abs_(at::Tensor & self)
{
    return op_api::abs_out(self, self);
}

at::Tensor acos(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnAcos, acl_op::acos(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAcos, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & acos_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAcos, acl_op::acos_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAcos, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & acos_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceAcos, acl_op::acos_(self));
    EXEC_NPU_CMD(aclnnInplaceAcos, self);
    return self;
}

at::Tensor acosh(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnAcosh, acl_op::acosh(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAcosh, self, out);
    return out;
}

at::Tensor & acosh_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAcosh, acl_op::acosh_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAcosh, self, out);
    return out;
}

at::Tensor & acosh_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceAcosh, acl_op::acosh_(self));
    EXEC_NPU_CMD(aclnnInplaceAcosh, self);
    return self;
}

at::Tensor & adaptive_avg_pool3d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAdaptiveAvgPool3d, acl_op::adaptive_avg_pool3d_out(self, output_size, out));
    auto output_size_0 = out.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAdaptiveAvgPool3d, self, output_size, out);
    return out;
}

at::Tensor & adaptive_avg_pool3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnAdaptiveAvgPool3dBackward, acl_op::adaptive_avg_pool3d_backward_out(grad_output, self, grad_input));
    auto output_size_0 = grad_input.sizes();
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAdaptiveAvgPool3dBackward, grad_output, self, grad_input);
    return grad_input;
}

::std::tuple<at::Tensor,at::Tensor> adaptive_max_pool2d(const at::Tensor & self, at::IntArrayRef output_size)
{
    DO_COMPATIBILITY(aclnnAdaptiveMaxPool2d, acl_op::adaptive_max_pool2d(self, output_size));
    auto output_size_0 = max_pool2d_out_size(self, output_size);
    auto output_dtype_0 = self.scalar_type();
    auto output_dtype_1 = at::kLong;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    at::Tensor indices = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnAdaptiveMaxPool2d, self, output_size, out, indices);
    return std::make_tuple(std::move(out), std::move(indices));
}

::std::tuple<at::Tensor &,at::Tensor &> adaptive_max_pool2d_out(const at::Tensor & self, at::IntArrayRef output_size, at::Tensor & out, at::Tensor & indices)
{
    DO_COMPATIBILITY(aclnnAdaptiveMaxPool2d, acl_op::adaptive_max_pool2d_out(self, output_size, out, indices));
    auto output_size_0 = max_pool2d_out_size(self, output_size);
    auto output_dtype_0 = self.scalar_type();
    auto output_dtype_1 = at::kLong;
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    npu_preparation::check_tensor({self}, indices, output_dtype_1, output_size_0);
    EXEC_NPU_CMD(aclnnAdaptiveMaxPool2d, self, output_size, out, indices);
    return std::forward_as_tuple(out, indices);
}

at::Tensor adaptive_max_pool2d_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices)
{
    DO_COMPATIBILITY(aclnnAdaptiveMaxPool2dBackward, acl_op::adaptive_max_pool2d_backward(grad_output, self, indices));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAdaptiveMaxPool2dBackward, grad_output, self, indices, out);
    return out;
}

at::Tensor & adaptive_max_pool2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & indices, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnAdaptiveMaxPool2dBackward, acl_op::adaptive_max_pool2d_backward_out(grad_output, self, indices, grad_input));
    auto output_size_0 = grad_input.sizes();
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self, indices}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAdaptiveMaxPool2dBackward, grad_output, self, indices, grad_input);
    return grad_input;
}

at::Tensor addr(const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha)
{
    DO_COMPATIBILITY(aclnnAddr, acl_op::addr(self, vec1, vec2, beta, alpha));
    auto output_size_0 = addr_npu_output_size(self, vec1, vec2);
    auto output_dtype_0 = at::native::result_type({self, vec1, vec2});
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAddr, self, vec1, vec2, beta, alpha, out);
    return out;
}

at::Tensor & addr_out(const at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAddr, acl_op::addr_out(self, vec1, vec2, beta, alpha, out));
    auto output_size_0 = addr_npu_output_size(self, vec1, vec2);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, vec1, vec2}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAddr, self, vec1, vec2, beta, alpha, out);
    return out;
}

at::Tensor & addr_(at::Tensor & self, const at::Tensor & vec1, const at::Tensor & vec2, const at::Scalar & beta, const at::Scalar & alpha)
{
    DO_COMPATIBILITY(aclnnInplaceAddr, acl_op::addr_(self, vec1, vec2, beta, alpha));
    EXEC_NPU_CMD(aclnnInplaceAddr, self, vec1, vec2, beta, alpha);
    return self;
}

at::Tensor amax(const at::Tensor & self, at::IntArrayRef dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnAmax, acl_op::amax(self, dim, keepdim));
    auto output_size_0 = reduce_ops_npu_output_size(self, dim, keepdim);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAmax, self, dim, keepdim, out);
    return out;
}

at::Tensor & amax_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAmax, acl_op::amax_out(self, dim, keepdim, out));
    auto output_size_0 = reduce_ops_npu_output_size(self, dim, keepdim);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAmax, self, dim, keepdim, out);
    return out;
}

at::Tensor amin(const at::Tensor & self, at::IntArrayRef dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnAmin, acl_op::amin(self, dim, keepdim));
    auto output_size_0 = reduce_ops_npu_output_size(self, dim, keepdim);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAmin, self, dim, keepdim, out);
    return out;
}

at::Tensor & amin_out(const at::Tensor & self, at::IntArrayRef dim, bool keepdim, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAmin, acl_op::amin_out(self, dim, keepdim, out));
    auto output_size_0 = reduce_ops_npu_output_size(self, dim, keepdim);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAmin, self, dim, keepdim, out);
    return out;
}

at::Tensor angle(const at::Tensor & self)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = angle_out_dtype(self);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAngleV2, self, out);
    return out;
}

at::Tensor & angle_out(const at::Tensor & self, at::Tensor & out)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = angle_out_dtype(self);
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAngleV2, self, out);
    return out;
}

at::Tensor asin(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnAsin, acl_op::asin(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAsin, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & asin_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAsin, acl_op::asin_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAsin, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & asin_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceAsin, acl_op::asin_(self));
    EXEC_NPU_CMD(aclnnInplaceAsin, self);
    return self;
}

at::Tensor asinh(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnAsinh, acl_op::asinh(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAsinh, self, out);
    return out;
}

at::Tensor & asinh_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAsinh, acl_op::asinh_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAsinh, self, out);
    return out;
}

at::Tensor & asinh_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceAsinh, acl_op::asinh_(self));
    EXEC_NPU_CMD(aclnnInplaceAsinh, self);
    return self;
}

at::Tensor atan(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnAtan, acl_op::atan(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAtan, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & atan_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAtan, acl_op::atan_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAtan, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & atan_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceAtan, acl_op::atan_(self));
    EXEC_NPU_CMD(aclnnInplaceAtan, self);
    return self;
}

at::Tensor atanh(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnAtanh, acl_op::atanh(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAtanh, self, out);
    return out;
}

at::Tensor & atanh_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnAtanh, acl_op::atanh_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnAtanh, self, out);
    return out;
}

at::Tensor & atanh_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceAtanh, acl_op::atanh_(self));
    EXEC_NPU_CMD(aclnnInplaceAtanh, self);
    return self;
}

at::Tensor batch_norm_backward_elemt(const at::Tensor & grad_out, const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & weight, const at::Tensor & sum_dy, const at::Tensor & sum_dy_xmu, const at::Tensor & count)
{
    DO_COMPATIBILITY(aclnnBatchNormElemtBackward, acl_op::batch_norm_backward_elemt(grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count));
    auto output_size_0 = input.sizes();
    auto output_dtype_0 = input.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_out.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnBatchNormElemtBackward, grad_out, input, mean, invstd, weight, sum_dy, sum_dy_xmu, count, out);
    return out;
}

::std::tuple<at::Tensor,at::Tensor> batch_norm_gather_stats_with_counts(const at::Tensor & input, const at::Tensor & mean, const at::Tensor & invstd, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, double momentum, double eps, const at::Tensor & counts)
{
    DO_COMPATIBILITY(aclnnBatchNormGatherStatsWithCounts, acl_op::batch_norm_gather_stats_with_counts(input, mean, invstd, running_mean, running_var, momentum, eps, counts));
    auto output_size_0 = {input.size(1)};
    auto output_dtype_0 = input.scalar_type() == mean.scalar_type() && input.scalar_type() == at::kHalf ? at::kHalf : at::kFloat;
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnBatchNormGatherStatsWithCounts, input, mean, invstd, running_mean, running_var, momentum, eps, counts, out0, out1);
    return std::make_tuple(std::move(out0), std::move(out1));
}

::std::tuple<at::Tensor,at::Tensor> batch_norm_stats(const at::Tensor & input, double eps)
{
    DO_COMPATIBILITY(aclnnBatchNormStats, acl_op::batch_norm_stats(input, eps));
    auto output_size_0 = {input.size(1)};
    auto output_dtype_0 = at::kFloat;
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnBatchNormStats, input, eps, out0, out1);
    return std::make_tuple(std::move(out0), std::move(out1));
}

at::Tensor binary_cross_entropy(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction)
{
    DO_COMPATIBILITY(aclnnBinaryCrossEntropy, acl_op::binary_cross_entropy(self, target, weight, reduction));
    auto output_size_0 = reduction == at::Reduction::None ? self.sizes() : at::ArrayRef<int64_t>();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnBinaryCrossEntropy, self, target, weight, reduction, out);
    return out;
}

at::Tensor & binary_cross_entropy_out(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnBinaryCrossEntropy, acl_op::binary_cross_entropy_out(self, target, weight, reduction, out));
    auto output_size_0 = out.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, target}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnBinaryCrossEntropy, self, target, weight, reduction, out);
    return out;
}

at::Tensor binary_cross_entropy_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction)
{
    DO_COMPATIBILITY(aclnnBinaryCrossEntropyBackward, acl_op::binary_cross_entropy_backward(grad_output, self, target, weight, reduction));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnBinaryCrossEntropyBackward, grad_output, self, target, weight, reduction, out);
    return out;
}

at::Tensor & binary_cross_entropy_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, int64_t reduction, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnBinaryCrossEntropyBackward, acl_op::binary_cross_entropy_backward_out(grad_output, self, target, weight, reduction, grad_input));
    auto output_size_0 = grad_input.sizes();
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self, target}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnBinaryCrossEntropyBackward, grad_output, self, target, weight, reduction, grad_input);
    return grad_input;
}

at::Tensor binary_cross_entropy_with_logits(const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & pos_weight, int64_t reduction)
{
    DO_COMPATIBILITY(aclnnBinaryCrossEntropyWithLogits, acl_op::binary_cross_entropy_with_logits(self, target, weight, pos_weight, reduction));
    auto output_size_0 = (reduction == at::Reduction::None) ? input_same_output_size(target) : at::ArrayRef<int64_t>();
    auto output_dtype_0 = target.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnBinaryCrossEntropyWithLogits, self, target, weight, pos_weight, reduction, out);
    return out;
}

at::Tensor bitwise_not(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnBitwiseNot, acl_op::bitwise_not(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnBitwiseNot, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & bitwise_not_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnBitwiseNot, acl_op::bitwise_not_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnBitwiseNot, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & bitwise_not_(at::Tensor & self)
{
    return op_api::bitwise_not_out(self, self);
}

at::Tensor cdist(const at::Tensor & x1, const at::Tensor & x2, double p, c10::optional<int64_t> compute_mode)
{
    DO_COMPATIBILITY(aclnnCdist, acl_op::cdist(x1, x2, p, compute_mode));
    auto p_float = static_cast<float>(p);
    auto mode = std::abs(p - 2.0) < 1e-6 ? 2 : compute_mode.value_or(0);
    auto output_size_0 = op_infer::cdist_npu_output_size(x1, x2);
    auto output_dtype_0 = x1.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x1.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCdist, x1, x2, p_float, mode, out);
    return out;
}

at::Tensor ceil(const at::Tensor & self)
{
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
        return acl_op::ceil(self);
    }
    DO_COMPATIBILITY(aclnnCeil, acl_op::ceil(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCeil, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & ceil_out(const at::Tensor & self, at::Tensor & out)
{
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
        return acl_op::ceil_out(self, out);
    }
    DO_COMPATIBILITY(aclnnCeil, acl_op::ceil_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnCeil, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & ceil_(at::Tensor & self)
{
    return op_api::ceil_out(self, self);
}

at::Tensor celu(const at::Tensor & self, const at::Scalar & alpha)
{
    DO_COMPATIBILITY(aclnnCelu, acl_op::celu(self, alpha));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCelu, self, alpha, out);
    return out;
}

at::Tensor & celu_(at::Tensor & self, const at::Scalar & alpha)
{
    DO_COMPATIBILITY(aclnnInplaceCelu, acl_op::celu_(self, alpha));
    EXEC_NPU_CMD(aclnnInplaceCelu, self, alpha);
    return self;
}

at::Tensor channel_shuffle(const at::Tensor & self, int64_t groups)
{
    DO_COMPATIBILITY(aclnnChannelShuffle, acl_op::channel_shuffle(self, groups));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnChannelShuffle, self, groups, out);
    return out;
}

at::Tensor clamp(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max)
{
    DO_COMPATIBILITY(aclnnClamp, acl_op::clamp(self, min, max));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = is_gte_cann_version_820rc1()?clamp_scalar_out_dtype(self, min, max):c10::typeMetaToScalarType(self.dtype());
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnClamp, self, min, max, out);
    return out;
}

at::Tensor clamp(const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max)
{
    DO_COMPATIBILITY(aclnnClampTensor, acl_op::clamp(self, min, max));
    auto output_size_0 = clamp_npu_output_size(self, min, max);
    auto output_dtype_0 = is_gte_cann_version_820rc1()?clamp_out_dtype(self, min, max):c10::typeMetaToScalarType(self.dtype());
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnClampTensor, self, min, max, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & clamp_out(const at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnClampTensor, acl_op::clamp_out(self, min, max, out));
    auto output_size_0 = clamp_npu_output_size(self, min, max);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnClampTensor, self, min, max, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & clamp_out(const at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnClamp, acl_op::clamp_out(self, min, max, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnClamp, self, min, max, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & clamp_(at::Tensor & self, const c10::optional<at::Scalar> & min, const c10::optional<at::Scalar> & max)
{
    return op_api::clamp_out(self, min, max, self);
}

at::Tensor & clamp_(at::Tensor & self, const c10::optional<at::Tensor> & min, const c10::optional<at::Tensor> & max)
{
    return op_api::clamp_out(self, min, max, self);
}

at::Tensor clamp_max(const at::Tensor & self, const at::Scalar & max)
{
    DO_COMPATIBILITY(aclnnClampMax, acl_op::clamp_max(self, max));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = is_gte_cann_version_820rc1()?at::native::result_type(self, max):c10::typeMetaToScalarType(self.dtype());
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnClampMax, self, max, out);
    return out;
}

at::Tensor clamp_max(const at::Tensor & self, const at::Tensor & max)
{
    DO_COMPATIBILITY(aclnnClampMaxTensor, acl_op::clamp_max(self, max));
    auto output_size_0 = op_infer::broadcast_ops_npu_output_size(self, max);
    auto output_dtype_0 = is_gte_cann_version_820rc1()?at::native::result_type(self, max):c10::typeMetaToScalarType(self.dtype());
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnClampMaxTensor, self, max, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & clamp_max_out(const at::Tensor & self, const at::Tensor & max, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnClampMaxTensor, acl_op::clamp_max_out(self, max, out));
    auto output_size_0 = op_infer::broadcast_ops_npu_output_size(self, max);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, max}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnClampMaxTensor, self, max, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & clamp_max_out(const at::Tensor & self, const at::Scalar & max, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnClampMax, acl_op::clamp_max_out(self, max, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnClampMax, self, max, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & clamp_max_(at::Tensor & self, const at::Scalar & max)
{
    DO_COMPATIBILITY(aclnnInplaceClampMax, acl_op::clamp_max_(self, max));
    EXEC_NPU_CMD(aclnnInplaceClampMax, self, max);
    return self;
}

at::Tensor & clamp_max_(at::Tensor & self, const at::Tensor & max)
{
    DO_COMPATIBILITY(aclnnInplaceClampMaxTensor, acl_op::clamp_max_(self, max));
    EXEC_NPU_CMD(aclnnInplaceClampMaxTensor, self, max);
    return self;
}

at::Tensor clamp_min(const at::Tensor & self, const at::Scalar & min)
{
    DO_COMPATIBILITY(aclnnClampMin, acl_op::clamp_min(self, min));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = is_gte_cann_version_820rc1()?at::native::result_type(self, min):c10::typeMetaToScalarType(self.dtype());
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnClampMin, self, min, out);
    return out;
}

at::Tensor clamp_min(const at::Tensor & self, const at::Tensor & min)
{
    DO_COMPATIBILITY(aclnnClampMinTensor, acl_op::clamp_min(self, min));
    auto output_size_0 = op_infer::broadcast_ops_npu_output_size(self, min);
    auto output_dtype_0 = is_gte_cann_version_820rc1()?at::native::result_type(self, min):c10::typeMetaToScalarType(self.dtype());
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnClampMinTensor, self, min, out);
    return out;
}

at::Tensor & clamp_min_out(const at::Tensor & self, const at::Tensor & min, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnClampMinTensor, acl_op::clamp_min_out(self, min, out));
    auto output_size_0 = op_infer::broadcast_ops_npu_output_size(self, min);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, min}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnClampMinTensor, self, min, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & clamp_min_out(const at::Tensor & self, const at::Scalar & min, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnClampMin, acl_op::clamp_min_out(self, min, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnClampMin, self, min, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & clamp_min_(at::Tensor & self, const at::Scalar & min)
{
    return op_api::clamp_min_out(self, min, self);
}

at::Tensor & clamp_min_(at::Tensor & self, const at::Tensor & min)
{
    DO_COMPATIBILITY(aclnnInplaceClampMinTensor, acl_op::clamp_min_(self, min));
    EXEC_NPU_CMD(aclnnInplaceClampMinTensor, self, min);
    return self;
}

at::Tensor conv_tbc(const at::Tensor & self, const at::Tensor & weight, const at::Tensor & bias, int64_t pad)
{
    DO_COMPATIBILITY(aclnnConvTbc, acl_op::conv_tbc(self, weight, bias, pad));
    auto cube_math_type = npu_preparation::get_cube_math_type(at_npu::native::env::IsAllowConvHF32());
    auto output_size_0 = {self.size(0) + 2 * pad - weight.size(0) + 1, self.size(1), weight.size(2)};
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnConvTbc, self, weight, bias, pad, out, cube_math_type);
    return out;
}

at::Tensor cos(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnCos, acl_op::cos(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor result = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCos, self, result);
    at::namedinference::propagate_names(result, self);
    return result;
}

at::Tensor & cos_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnCos, acl_op::cos_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnCos, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & cos_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceCos, acl_op::cos_(self));
    EXEC_NPU_CMD(aclnnInplaceCos, self);
    return self;
}

at::Tensor cosh(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnCosh, acl_op::cosh(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCosh, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & cosh_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnCosh, acl_op::cosh_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnCosh, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & cosh_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceCosh, acl_op::cosh_(self));
    EXEC_NPU_CMD(aclnnInplaceCosh, self);
    return self;
}

at::Tensor col2im(const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride)
{
    DO_COMPATIBILITY(aclnnIm2colBackward, acl_op::col2im(self, output_size, kernel_size, dilation, padding, stride));
    auto output_size_0 = im2col_backward_npu_output_size(self, output_size, kernel_size);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnIm2colBackward, self, output_size, kernel_size, dilation, padding, stride, out);
    return out;
}

at::Tensor & col2im_out(const at::Tensor & self, at::IntArrayRef output_size, at::IntArrayRef kernel_size, at::IntArrayRef dilation, at::IntArrayRef padding, at::IntArrayRef stride, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnIm2colBackward, acl_op::col2im_out(self, output_size, kernel_size, dilation, padding, stride, out));
    auto output_size_0 = im2col_backward_npu_output_size(self, output_size, kernel_size);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnIm2colBackward, self, output_size, kernel_size, dilation, padding, stride, out);
    return out;
}

at::Tensor & cumprod_out(const at::Tensor & self, int64_t dim, c10::optional<at::ScalarType> dtype, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnCumprod, acl_op::cumprod_out(self, dim, dtype, out));
    auto dim_scalar = at::Scalar(dim);
    auto dst_type = dtype.has_value() ? dtype.value() : out.scalar_type();
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = dtype.has_value() ? dtype.value() : out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnCumprod, self, dim_scalar, dst_type, out);
    return out;
}

at::Tensor dot(const at::Tensor & self, const at::Tensor & tensor)
{
    DO_COMPATIBILITY(aclnnDot, acl_op::dot(self, tensor));
    auto output_size_0 = c10::SmallVector<int64_t, op_infer::SIZE>{};
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnDot, self, tensor, out);
    return out;
}

at::Tensor & dot_out(const at::Tensor & self, const at::Tensor & tensor, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnDot, acl_op::dot_out(self, tensor, out));
    auto output_size_0 = c10::SmallVector<int64_t, op_infer::SIZE>{};
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, tensor}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnDot, self, tensor, out);
    return out;
}

at::Tensor elu(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale)
{
    DO_COMPATIBILITY(aclnnElu, acl_op::elu(self, alpha, scale, input_scale));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnElu, self, alpha, scale, input_scale, out);
    return out;
}

at::Tensor & elu_out(const at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnElu, acl_op::elu_out(self, alpha, scale, input_scale, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnElu, self, alpha, scale, input_scale, out);
    return out;
}

at::Tensor & elu_(at::Tensor & self, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale)
{
    DO_COMPATIBILITY(aclnnInplaceElu, acl_op::elu_(self, alpha, scale, input_scale));
    EXEC_NPU_CMD(aclnnInplaceElu, self, alpha, scale, input_scale);
    return self;
}

at::Tensor elu_backward(const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result)
{
    DO_COMPATIBILITY(aclnnEluBackward, acl_op::elu_backward(grad_output, alpha, scale, input_scale, is_result, self_or_result));
    auto output_size_0 = grad_output.sizes();
    auto output_dtype_0 = grad_output.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnEluBackward, grad_output, alpha, scale, input_scale, is_result, self_or_result, out);
    return out;
}

at::Tensor & elu_backward_out(const at::Tensor & grad_output, const at::Scalar & alpha, const at::Scalar & scale, const at::Scalar & input_scale, bool is_result, const at::Tensor & self_or_result, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnEluBackward, acl_op::elu_backward_out(grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input));
    auto output_size_0 = grad_output.sizes();
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self_or_result}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnEluBackward, grad_output, alpha, scale, input_scale, is_result, self_or_result, grad_input);
    return grad_input;
}

at::Tensor embedding_dense_backward(const at::Tensor & grad_output, const at::Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq)
{
    DO_COMPATIBILITY(aclnnEmbeddingDenseBackward, acl_op::embedding_dense_backward(grad_output, indices, num_weights, padding_idx, scale_grad_by_freq));
    auto output_size_0 = {num_weights, grad_output.size(-1)};
    auto output_dtype_0 = grad_output.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnEmbeddingDenseBackward, grad_output, indices, num_weights, padding_idx, scale_grad_by_freq, out);
    return out;
}

at::Tensor erf(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnErf, acl_op::erf(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = (self.scalar_type() == at::ScalarType::Bool || self.scalar_type() == at::ScalarType::Long || self.scalar_type() == at::ScalarType::Int) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnErf, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & erf_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnErf, acl_op::erf_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnErf, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & erf_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceErf, acl_op::erf_(self));
    EXEC_NPU_CMD(aclnnInplaceErf, self);
    return self;
}

at::Tensor erfc(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnErfc, acl_op::erfc(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = (self.scalar_type() == at::ScalarType::Bool || self.scalar_type() == at::ScalarType::Long || self.scalar_type() == at::ScalarType::Int) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnErfc, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & erfc_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnErfc, acl_op::erfc_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnErfc, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & erfc_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceErfc, acl_op::erfc_(self));
    EXEC_NPU_CMD(aclnnInplaceErfc, self);
    return self;
}

at::Tensor erfinv(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnErfinv, acl_op::erfinv(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnErfinv, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & erfinv_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnErfinv, acl_op::erfinv_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnErfinv, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & erfinv_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceErfinv, acl_op::erfinv_(self));
    EXEC_NPU_CMD(aclnnInplaceErfinv, self);
    return self;
}

at::Tensor exp(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnExp, acl_op::exp(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = (self.scalar_type() == at::ScalarType::Bool || self.scalar_type() == at::ScalarType::Long) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnExp, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & exp_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnExp, acl_op::exp_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnExp, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor exp2(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnExp2, acl_op::exp2(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::ScalarType::Float : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnExp2, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & exp2_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnExp2, acl_op::exp2_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnExp2, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & exp2_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceExp2, acl_op::exp2_(self));
    EXEC_NPU_CMD(aclnnInplaceExp2, self);
    return self;
}

at::Tensor & exp_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceExp, acl_op::exp_(self));
    EXEC_NPU_CMD(aclnnInplaceExp, self);
    return self;
}

at::Tensor expm1(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnExpm1, acl_op::expm1(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isFloatingType(self.scalar_type()) ? self.scalar_type() : at::kFloat;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnExpm1, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & expm1_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnExpm1, acl_op::expm1_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnExpm1, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & expm1_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceExpm1, acl_op::expm1_(self));
    EXEC_NPU_CMD(aclnnInplaceExpm1, self);
    return self;
}

at::Tensor & fill_diagonal_(at::Tensor & self, const at::Scalar & fill_value, bool wrap)
{
    DO_COMPATIBILITY(aclnnInplaceFillDiagonal, acl_op::fill_diagonal_(self, fill_value, wrap));
    EXEC_NPU_CMD(aclnnInplaceFillDiagonal, self, fill_value, wrap);
    return self;
}

at::Tensor flip(const at::Tensor & self, at::IntArrayRef dims)
{
    DO_COMPATIBILITY(aclnnFlip, acl_op::flip(self, dims));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnFlip, self, dims, out);
    return out;
}

at::Tensor floor(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnFloor, acl_op::floor(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnFloor, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & floor_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnFloor, acl_op::floor_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnFloor, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & floor_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceFloor, acl_op::floor_(self));
    EXEC_NPU_CMD(aclnnInplaceFloor, self);
    return self;
}

at::Tensor fmod(const at::Tensor & self, const at::Scalar & other)
{
    DO_COMPATIBILITY(aclnnFmodScalar, acl_op::fmod(self, other));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = at::native::result_type(self, other);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnFmodScalar, self, other, out);
    return out;
}

at::Tensor & fmod_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnFmodScalar, acl_op::fmod_out(self, other, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnFmodScalar, self, other, out);
    return out;
}

at::Tensor & fmod_(at::Tensor & self, const at::Scalar & other)
{
    DO_COMPATIBILITY(aclnnInplaceFmodScalar, acl_op::fmod_(self, other));
    EXEC_NPU_CMD(aclnnInplaceFmodScalar, self, other);
    return self;
}

at::Tensor frac(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnFrac, acl_op::frac(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnFrac, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & frac_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnFrac, acl_op::frac_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnFrac, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & frac_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceFrac, acl_op::frac_(self));
    EXEC_NPU_CMD(aclnnInplaceFrac, self);
    return self;
}

at::Tensor & gcd_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnGcd, acl_op::gcd_out(self, other, out));
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, other}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnGcd, self, other, out);
    return out;
}

at::Tensor glu(const at::Tensor & self, int64_t dim)
{
    DO_COMPATIBILITY(aclnnGlu, acl_op::glu(self, dim));
    auto output_size_0 = glu_npu_output_size(self, dim);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGlu, self, dim, out);
    return out;
}

at::Tensor & glu_out(const at::Tensor & self, int64_t dim, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnGlu, acl_op::glu_out(self, dim, out));
    auto output_size_0 = glu_npu_output_size(self, dim);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnGlu, self, dim, out);
    return out;
}

at::Tensor glu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim)
{
    DO_COMPATIBILITY(aclnnGluBackward, acl_op::glu_backward(grad_output, self, dim));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGluBackward, grad_output, self, dim, out);
    return out;
}

at::Tensor & glu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnGluBackward, acl_op::glu_backward_out(grad_output, self, dim, grad_input));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = grad_output.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnGluBackward, grad_output, self, dim, grad_input);
    return grad_input;
}

at::Tensor grid_sampler_2d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)
{
    DO_COMPATIBILITY(aclnnGridSampler2D, acl_op::grid_sampler_2d(input, grid, interpolation_mode, padding_mode, align_corners));
    auto output_size_0 = {input.size(0), input.size(1), grid.size(1), grid.size(2)};
    auto output_dtype_0 = input.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGridSampler2D, input, grid, interpolation_mode, padding_mode, align_corners, out0);
    return out0;
}

::std::tuple<at::Tensor,at::Tensor> grid_sampler_2d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask)
{
    DO_COMPATIBILITY(aclnnGridSampler2DBackward, acl_op::grid_sampler_2d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask));
    auto output_size_0 = input.sizes();
    auto output_size_1 = grid.sizes();
    auto output_dtype_0 = input.scalar_type();
    auto output_dtype_1 = grid.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                grad_output.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnGridSampler2DBackward, grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask, out0, out1);
    return std::make_tuple(std::move(out0), std::move(out1));
}

at::Tensor grid_sampler_3d(const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners)
{
    DO_COMPATIBILITY(aclnnGridSampler3D, acl_op::grid_sampler_3d(input, grid, interpolation_mode, padding_mode, align_corners));
    auto output_size_0 = {input.size(0), input.size(1), grid.size(1), grid.size(2), grid.size(3)};
    auto output_dtype_0 = input.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGridSampler3D, input, grid, interpolation_mode, padding_mode, align_corners, out0);
    return out0;
}

::std::tuple<at::Tensor,at::Tensor> grid_sampler_3d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & grid, int64_t interpolation_mode, int64_t padding_mode, bool align_corners, ::std::array<bool,2> output_mask)
{
    DO_COMPATIBILITY(aclnnGridSampler3DBackward, acl_op::grid_sampler_3d_backward(grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask));
    auto output_size_0 = input.sizes();
    auto output_size_1 = grid.sizes();
    auto output_dtype_0 = input.scalar_type();
    auto output_dtype_1 = grid.scalar_type();
    at::Tensor dinput = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    at::Tensor dgrid = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                grad_output.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnGridSampler3DBackward, grad_output, input, grid, interpolation_mode, padding_mode, align_corners, output_mask, dinput, dgrid);
    return std::make_tuple(std::move(dinput), std::move(dgrid));
}

at::Tensor hardshrink(const at::Tensor & self, const at::Scalar & lambd)
{
    DO_COMPATIBILITY(aclnnHardshrink, acl_op::hardshrink(self, lambd));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnHardshrink, self, lambd, out);
    return out;
}

at::Tensor & hardshrink_out(const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnHardshrink, acl_op::hardshrink_out(self, lambd, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnHardshrink, self, lambd, out);
    return out;
}

at::Tensor hardshrink_backward(const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd)
{
    DO_COMPATIBILITY(aclnnHardshrinkBackward, acl_op::hardshrink_backward(grad_out, self, lambd));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_out, self);
    auto output_dtype_0 = at::native::result_type(grad_out, self);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_out.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnHardshrinkBackward, grad_out, self, lambd, grad_input);
    return grad_input;
}

at::Tensor & hardshrink_backward_out(const at::Tensor & grad_out, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnHardshrinkBackward, acl_op::hardshrink_backward_out(grad_out, self, lambd, grad_input));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_out, self);
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_out, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnHardshrinkBackward, grad_out, self, lambd, grad_input);
    return grad_input;
}

at::Tensor hardsigmoid(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnHardsigmoid, acl_op::hardsigmoid(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnHardsigmoid, self, out);
    return out;
}

at::Tensor & hardsigmoid_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnHardsigmoid, acl_op::hardsigmoid_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnHardsigmoid, self, out);
    return out;
}

at::Tensor & hardsigmoid_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceHardsigmoid, acl_op::hardsigmoid_(self));
    EXEC_NPU_CMD(aclnnInplaceHardsigmoid, self);
    return self;
}

at::Tensor hardsigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnHardsigmoidBackward, acl_op::hardsigmoid_backward(grad_output, self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = at::native::result_type(grad_output, self);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnHardsigmoidBackward, grad_output, self, out);
    return out;
}

at::Tensor hardswish(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnHardswish, acl_op::hardswish(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnHardswish, self, out);
    return out;
}

at::Tensor & hardswish_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnHardswish, acl_op::hardswish_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnHardswish, self, out);
    return out;
}

at::Tensor & hardswish_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceHardswish, acl_op::hardswish_(self));
    EXEC_NPU_CMD(aclnnInplaceHardswish, self);
    return self;
}

at::Tensor hardtanh(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val)
{
    DO_COMPATIBILITY(aclnnHardtanh, acl_op::hardtanh(self, min_val, max_val));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnHardtanh, self, min_val, max_val, out);
    return out;
}

at::Tensor & hardtanh_out(const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnHardtanh, acl_op::hardtanh_out(self, min_val, max_val, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnHardtanh, self, min_val, max_val, out);
    return out;
}

at::Tensor & hardtanh_(at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val)
{
    DO_COMPATIBILITY(aclnnInplaceHardtanh, acl_op::hardtanh_(self, min_val, max_val));
    EXEC_NPU_CMD(aclnnInplaceHardtanh, self, min_val, max_val);
    return self;
}

at::Tensor hardtanh_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val)
{
    DO_COMPATIBILITY(aclnnHardtanhBackward, acl_op::hardtanh_backward(grad_output, self, min_val, max_val));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnHardtanhBackward, grad_output, self, min_val, max_val, grad_input);
    return grad_input;
}

at::Tensor & hardtanh_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & min_val, const at::Scalar & max_val, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnHardtanhBackward, acl_op::hardtanh_backward_out(grad_output, self, min_val, max_val, grad_input));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnHardtanhBackward, grad_output, self, min_val, max_val, grad_input);
    return grad_input;
}

at::Tensor index_copy(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source)
{
    DO_COMPATIBILITY(aclnnIndexCopy, acl_op::index_copy(self, dim, index, source));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnIndexCopy, self, dim, index, source, out);
    return out;
}

at::Tensor & index_copy_out(const at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source, at::Tensor & out)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self, index, source}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnIndexCopy, self, dim, index, source, out);
    return out;
}

at::Tensor & index_copy_(at::Tensor & self, int64_t dim, const at::Tensor & index, const at::Tensor & source)
{
    DO_COMPATIBILITY(aclnnInplaceIndexCopy, acl_op::index_copy_(self, dim, index, source));
    EXEC_NPU_CMD(aclnnInplaceIndexCopy, self, dim, index, source);
    return self;
}

at::Tensor inverse(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInverse, acl_op::inverse(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnInverse, self, out);
    return out;
}

at::Tensor & inverse_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnInverse, acl_op::inverse_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnInverse, self, out);
    return out;
}

at::Tensor isclose(const at::Tensor & self, const at::Tensor & other, double rtol, double atol, bool equal_nan)
{
    DO_COMPATIBILITY(aclnnIsClose, acl_op::isclose(self, other, rtol, atol, equal_nan));
    auto output_size_0 = isclose_output_size(self, other, rtol, atol, equal_nan);
    auto output_dtype_0 = at::kBool;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnIsClose, self, other, rtol, atol, equal_nan, out);
    return out;
}

at::Tensor isfinite(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnIsFinite, acl_op::isfinite(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = at::kBool;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnIsFinite, self, out);
    return out;
}

at::Tensor & isin_out(const at::Scalar & element, const at::Tensor & test_elements, bool assume_unique, bool invert, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnIsInScalarTensor, acl_op::isin_out(element, test_elements, assume_unique, invert, out));
    auto output_size_0 = c10::SmallVector<int64_t, op_infer::SIZE>{};
    auto output_dtype_0 = at::kBool;
    npu_preparation::check_tensor({test_elements}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnIsInScalarTensor, element, test_elements, assume_unique, invert, out);
    return out;
}

at::Tensor isin(const at::Tensor & element, const at::Scalar & test_elements, bool assume_unique, bool invert)
{
    DO_COMPATIBILITY(aclnnIsInTensorScalar, acl_op::isin(element, test_elements, assume_unique, invert));
    auto output_size_0 = element.sizes();
    auto output_dtype_0 = at::kBool;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                element.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnIsInTensorScalar, element, test_elements, assume_unique, invert, out);
    return out;
}

at::Tensor & isin_out(const at::Tensor & element, const at::Scalar & test_elements, bool assume_unique, bool invert, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnIsInTensorScalar, acl_op::isin_out(element, test_elements, assume_unique, invert, out));
    auto output_size_0 = element.sizes();
    auto output_dtype_0 = at::kBool;
    npu_preparation::check_tensor({element}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnIsInTensorScalar, element, test_elements, assume_unique, invert, out);
    return out;
}

at::Tensor & isneginf_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnIsNegInf, acl_op::isneginf_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnIsNegInf, self, out);
    return out;
}

at::Tensor & isposinf_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnIsPosInf, acl_op::isposinf_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnIsPosInf, self, out);
    return out;
}

at::Tensor kl_div(const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target)
{
    DO_COMPATIBILITY(aclnnKlDiv, acl_op::kl_div(self, target, reduction, log_target));
    auto output_size_0 = reduction == at::Reduction::None ? broadcast_ops_npu_output_size(self.sizes(), target.sizes()) : at::ArrayRef<int64_t>();
    auto output_dtype_0 = at::native::result_type(self, target);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnKlDiv, self, target, reduction, log_target, out);
    return out;
}

at::Tensor l1_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction)
{
    DO_COMPATIBILITY(aclnnL1Loss, acl_op::l1_loss(self, target, reduction));
    auto output_size_0 = (reduction == at::Reduction::None) ? broadcast_ops_npu_output_size(self, target) : c10::SmallVector<int64_t, op_infer::SIZE>{};
    auto output_dtype_0 = at::native::result_type(target, self);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnL1Loss, self, target, reduction, out);
    return out;
}

at::Tensor leaky_relu(const at::Tensor & self, const at::Scalar & negative_slope)
{
    DO_COMPATIBILITY(aclnnLeakyRelu, acl_op::leaky_relu(self, negative_slope));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLeakyRelu, self, negative_slope, out);
    return out;
}

at::Tensor & leaky_relu_out(const at::Tensor & self, const at::Scalar & negative_slope, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLeakyRelu, acl_op::leaky_relu_out(self, negative_slope, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLeakyRelu, self, negative_slope, out);
    return out;
}

at::Tensor & leaky_relu_(at::Tensor & self, const at::Scalar & negative_slope)
{
    DO_COMPATIBILITY(aclnnInplaceLeakyRelu, acl_op::leaky_relu_(self, negative_slope));
    EXEC_NPU_CMD(aclnnInplaceLeakyRelu, self, negative_slope);
    return self;
}

at::Tensor leaky_relu_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result)
{
    DO_COMPATIBILITY(aclnnLeakyReluBackward, acl_op::leaky_relu_backward(grad_output, self, negative_slope, self_is_result));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLeakyReluBackward, grad_output, self, negative_slope, self_is_result, out);
    return out;
}

at::Tensor & leaky_relu_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & negative_slope, bool self_is_result, at::Tensor & grad_input)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLeakyReluBackward, grad_output, self, negative_slope, self_is_result, grad_input);
    return grad_input;
}

at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight)
{
    DO_COMPATIBILITY(aclnnLerps, acl_op::lerp(self, end, weight));
    auto output_size_0 = broadcast_ops_npu_output_size(self, end);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLerps, self, end, weight, out);
    return out;
}

at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const at::Scalar & weight, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLerps, acl_op::lerp_out(self, end, weight, out));
    auto output_size_0 = broadcast_ops_npu_output_size(self, end);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, end}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLerps, self, end, weight, out);
    return out;
}

at::Tensor lerp(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight)
{
    DO_COMPATIBILITY(aclnnLerp, acl_op::lerp(self, end, weight));
    DO_COMPATIBILITY(aclnnLerps, acl_op::lerp(self, end, weight));
    auto output_size_0 = broadcast_ops_npu_output_size(broadcast_ops_npu_output_size(self, end), weight.sizes());
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    if (npu_preparation::IsCPUScalar(weight)) {
        at::Scalar weight_scalar = weight.item();
        EXEC_NPU_CMD(aclnnLerps, self, end, weight_scalar, out);
        return out;
    }
    EXEC_NPU_CMD(aclnnLerp, self, end, weight, out);
    return out;
}

at::Tensor & lerp_out(const at::Tensor & self, const at::Tensor & end, const at::Tensor & weight, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLerp, acl_op::lerp_out(self, end, weight, out));
    DO_COMPATIBILITY(aclnnLerps, acl_op::lerp_out(self, end, weight, out));
    auto output_size_0 = broadcast_ops_npu_output_size(broadcast_ops_npu_output_size(self, end), weight.sizes());
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, end, weight}, out, output_dtype_0, output_size_0);
    if (npu_preparation::IsCPUScalar(weight)) {
        at::Scalar weight_scalar = weight.item();
        EXEC_NPU_CMD(aclnnLerps, self, end, weight_scalar, out);
        return out;
    }
    EXEC_NPU_CMD(aclnnLerp, self, end, weight, out);
    return out;
}

at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Scalar & weight)
{
    DO_COMPATIBILITY(aclnnInplaceLerps, acl_op::lerp_(self, end, weight));
    EXEC_NPU_CMD(aclnnInplaceLerps, self, end, weight);
    return self;
}

at::Tensor & lerp_(at::Tensor & self, const at::Tensor & end, const at::Tensor & weight)
{
    DO_COMPATIBILITY(aclnnInplaceLerp, acl_op::lerp_(self, end, weight));
    DO_COMPATIBILITY(aclnnInplaceLerps, acl_op::lerp_(self, end, weight));
    if (npu_preparation::IsCPUScalar(weight)) {
        at::Scalar weight_scalar = weight.item();
        EXEC_NPU_CMD(aclnnInplaceLerps, self, end, weight_scalar);
        return self;
    }
    EXEC_NPU_CMD(aclnnInplaceLerp, self, end, weight);
    return self;
}

at::Tensor log10(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnLog10, acl_op::log10(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLog10, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & log10_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLog10, acl_op::log10_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLog10, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & log10_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceLog10, acl_op::log10_(self));
    EXEC_NPU_CMD(aclnnInplaceLog10, self);
    return self;
}

at::Tensor log1p(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnLog1p, acl_op::log1p(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLog1p, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & log1p_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLog1p, acl_op::log1p_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLog1p, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & log1p_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceLog1p, acl_op::log1p_(self));
    EXEC_NPU_CMD(aclnnInplaceLog1p, self);
    return self;
}

at::Tensor log2(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnLog2, acl_op::log2(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLog2, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & log2_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLog2, acl_op::log2_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLog2, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & log2_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceLog2, acl_op::log2_(self));
    EXEC_NPU_CMD(aclnnInplaceLog2, self);
    return self;
}

at::Tensor log_sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer)
{
    DO_COMPATIBILITY(aclnnLogSigmoidBackward, acl_op::log_sigmoid_backward(grad_output, self, buffer));
    auto output_size_0 = grad_output.sizes();
    auto output_dtype_0 = grad_output.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLogSigmoidBackward, grad_output, self, buffer, grad_input);
    return grad_input;
}

at::Tensor & log_sigmoid_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & buffer, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnLogSigmoidBackward, acl_op::log_sigmoid_backward_out(grad_output, self, buffer, grad_input));
    auto output_size_0 = grad_output.sizes();
    auto output_dtype_0 = grad_output.scalar_type();
    npu_preparation::check_tensor({grad_output, self, buffer}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLogSigmoidBackward, grad_output, self, buffer, grad_input);
    return grad_input;
}

at::Tensor logaddexp(const at::Tensor & self, const at::Tensor & other)
{
    DO_COMPATIBILITY(aclnnLogAddExp, acl_op::logaddexp(self, other));
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = at::native::result_type(self, other);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLogAddExp, self, other, out);
    return out;
}

at::Tensor & logaddexp_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLogAddExp, acl_op::logaddexp_out(self, other, out));
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, other}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLogAddExp, self, other, out);
    return out;
}

at::Tensor logaddexp2(const at::Tensor & self, const at::Tensor & other)
{
    DO_COMPATIBILITY(aclnnLogAddExp2, acl_op::logaddexp2(self, other));
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = at::native::result_type(self, other);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLogAddExp2, self, other, out);
    return out;
}

at::Tensor & logaddexp2_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLogAddExp2, acl_op::logaddexp2_out(self, other, out));
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, other}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLogAddExp2, self, other, out);
    return out;
}

at::Tensor logical_and(const at::Tensor & self, const at::Tensor & other)
{
    DO_COMPATIBILITY(aclnnLogicalAnd, acl_op::logical_and(self, other));
    at::Tensor other_cp = other;
    if (npu_preparation::IsCPUScalar(other)) {
        at::Scalar other_scalar = other.item();
        other_cp = npu_preparation::copy_scalar_to_device(other_scalar, other.scalar_type(), self.device());
    }
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    auto output_size_0 = broadcast_ops_npu_output_size(self, other_cp);
    auto output_dtype_0 = at::kBool;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLogicalAnd, self, other_cp, out);
    at::namedinference::propagate_names_if_nonempty(out, maybe_names);
    return out;
}

at::Tensor & logical_and_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLogicalAnd, acl_op::logical_and_out(self, other, out));
    at::Tensor other_cp = other;
    if (npu_preparation::IsCPUScalar(other)) {
        at::Scalar other_scalar = other.item();
        other_cp = npu_preparation::copy_scalar_to_device(other_scalar, other.scalar_type(), self.device());
    }
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    auto output_size_0 = broadcast_ops_npu_output_size(self, other_cp);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, other_cp}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLogicalAnd, self, other_cp, out);
    at::namedinference::propagate_names_if_nonempty(out, maybe_names);
    return out;
}

at::Tensor & logical_and_(at::Tensor & self, const at::Tensor & other)
{
    DO_COMPATIBILITY(aclnnInplaceLogicalAnd, acl_op::logical_and_(self, other));
    at::Tensor other_cp = other;
    if (npu_preparation::IsCPUScalar(other)) {
        at::Scalar other_scalar = other.item();
        other_cp = npu_preparation::copy_scalar_to_device(other_scalar, other.scalar_type(), self.device());
    }
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    EXEC_NPU_CMD(aclnnInplaceLogicalAnd, self, other_cp);
    at::namedinference::propagate_names_if_nonempty(self, maybe_names);
    return self;
}

at::Tensor logical_not(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnLogicalNot, acl_op::logical_not(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = at::kBool;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLogicalNot, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & logical_not_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLogicalNot, acl_op::logical_not_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLogicalNot, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & logical_not_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceLogicalNot, acl_op::logical_not_(self));
    EXEC_NPU_CMD(aclnnInplaceLogicalNot, self);
    return self;
}

at::Tensor logical_xor(const at::Tensor & self, const at::Tensor & other)
{
    DO_COMPATIBILITY(aclnnLogicalXor, acl_op::logical_xor(self, other));
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = at::kBool;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnLogicalXor, self, other, out);
    at::namedinference::propagate_names_if_nonempty(out, maybe_names);
    return out;
}

at::Tensor & logical_xor_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnLogicalXor, acl_op::logical_xor_out(self, other, out));
    std::vector<at::Tensor> tensor_list = {self, other};
    auto maybe_names = op_plugin::utils::compute_names_npu(tensor_list);
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, other}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnLogicalXor, self, other, out);
    at::namedinference::propagate_names_if_nonempty(out, maybe_names);
    return out;
}

at::Tensor max_unpool2d(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size)
{
    DO_COMPATIBILITY(aclnnMaxUnpool2d, acl_op::max_unpool2d(self, indices, output_size));
    auto output_size_0 = max_pool2d_out_size(self, output_size);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMaxUnpool2d, self, indices, output_size, out);
    return out;
}

at::Tensor & max_unpool2d_out(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnMaxUnpool2d, acl_op::max_unpool2d_out(self, indices, output_size, out));
    auto output_size_0 = max_pool2d_out_size(self, output_size);
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self, indices}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnMaxUnpool2d, self, indices, output_size, out);
    return out;
}

at::Tensor max_unpool3d(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding)
{
    DO_COMPATIBILITY(aclnnMaxUnpool3d, acl_op::max_unpool3d(self, indices, output_size, stride, padding));
    auto output_size_0 = max_pool3d_output_size(self, output_size);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMaxUnpool3d, self, indices, output_size, stride, padding, out);
    return out;
}

at::Tensor & max_unpool3d_out(const at::Tensor & self, const at::Tensor & indices, at::IntArrayRef output_size, at::IntArrayRef stride, at::IntArrayRef padding, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnMaxUnpool3d, acl_op::max_unpool3d_out(self, indices, output_size, stride, padding, out));
    auto output_size_0 = max_pool3d_output_size(self, output_size);
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self, indices}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnMaxUnpool3d, self, indices, output_size, stride, padding, out);
    return out;
}

at::Tensor median(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnMedian, acl_op::median(self));
    auto output_size_0 = reduce_ops_npu_output_size(self, get_dimlist_for_tensor(self), false);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMedian, self, out);
    return out;
}

::std::tuple<at::Tensor,at::Tensor> median(const at::Tensor & self, int64_t dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnMedianDim, acl_op::median(self, dim, keepdim));
    auto output_size_0 = reduce_ops_npu_output_size(self, {dim}, keepdim);
    auto output_dtype_0 = self.scalar_type();
    auto output_dtype_1 = at::kLong;
    at::Tensor values = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    at::Tensor indices = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnMedianDim, self, dim, keepdim, values, indices);
    return std::make_tuple(std::move(values), std::move(indices));
}

::std::tuple<at::Tensor &,at::Tensor &> median_out(const at::Tensor & self, int64_t dim, bool keepdim, at::Tensor & values, at::Tensor & indices)
{
    DO_COMPATIBILITY(aclnnMedianDim, acl_op::median_out(self, dim, keepdim, values, indices));
    auto output_size_0 = reduce_ops_npu_output_size(self, {dim}, keepdim);
    auto output_dtype_0 = values.scalar_type();
    auto output_dtype_1 = indices.scalar_type();
    npu_preparation::check_tensor({self}, values, output_dtype_0, output_size_0);
    npu_preparation::check_tensor({self}, indices, output_dtype_1, output_size_0);
    EXEC_NPU_CMD(aclnnMedianDim, self, dim, keepdim, values, indices);
    return std::forward_as_tuple(values, indices);
}

at::Tensor mish_backward(const at::Tensor & grad_output, const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnMishBackward, acl_op::mish_backward(grad_output, self));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_output, self);
    auto output_dtype_0 = at::native::result_type(grad_output, self);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMishBackward, grad_output, self, out);
    return out;
}

at::Tensor mse_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction)
{
    DO_COMPATIBILITY(aclnnMseLossBackward, acl_op::mse_loss_backward(grad_output, self, target, reduction));
    auto output_size_0 = broadcast_ops_npu_output_size(broadcast_ops_npu_output_size(grad_output, self), target.sizes());
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMseLossBackward, grad_output, self, target, reduction, out);
    return out;
}

at::Tensor & mse_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnMseLossBackward, acl_op::mse_loss_backward_out(grad_output, self, target, reduction, grad_input));
    auto output_size_0 = broadcast_ops_npu_output_size(broadcast_ops_npu_output_size(grad_output, self), target.sizes());
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self, target}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnMseLossBackward, grad_output, self, target, reduction, grad_input);
    return grad_input;
}

at::Tensor nanmedian(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnNanMedian, acl_op::nanmedian(self));
    auto output_size_0 = reduce_ops_npu_output_size(self, get_dimlist_for_tensor(self), false);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnNanMedian, self, out);
    return out;
}

::std::tuple<at::Tensor,at::Tensor> nanmedian(const at::Tensor & self, int64_t dim, bool keepdim)
{
    DO_COMPATIBILITY(aclnnNanMedianDim, acl_op::nanmedian(self, dim, keepdim));
    auto output_size_0 = reduce_ops_npu_output_size(self, dim, keepdim);
    auto output_dtype_0 = self.scalar_type();
    auto output_dtype_1 = at::kLong;
    at::Tensor output = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    at::Tensor indices = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnNanMedianDim, self, dim, keepdim, output, indices);
    return std::make_tuple(std::move(output), std::move(indices));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_batch_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps)
{
    DO_COMPATIBILITY(aclnnBatchNorm, acl_op::native_batch_norm(input, weight, bias, running_mean, running_var, training, momentum, eps));
    auto output_size_0 = input.sizes();
    auto output_size_1 = c10::SmallVector<int64_t, op_infer::SIZE>{input.size(1)};
    auto output_dtype_0 = input.scalar_type();
    auto output_dtype_1 = training ? at::kFloat : input.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                input.options().dtype(output_dtype_1));
    at::Tensor out2 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                input.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnBatchNorm, input, weight, bias, running_mean, running_var, training, momentum, eps, out0, out1, out2);
    return std::make_tuple(std::move(out0), std::move(out1), std::move(out2));
}

::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &> native_batch_norm_out(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & running_mean, const c10::optional<at::Tensor> & running_var, bool training, double momentum, double eps, at::Tensor & out, at::Tensor & save_mean, at::Tensor & save_invstd)
{
    DO_COMPATIBILITY(aclnnBatchNorm, acl_op::native_batch_norm_out(input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd));
    auto output_size_0 = out.sizes();
    auto output_size_1 = save_mean.sizes();
    auto output_size_2 = save_invstd.sizes();
    auto output_dtype_0 = out.scalar_type();
    auto output_dtype_1 = save_mean.scalar_type();
    auto output_dtype_2 = save_invstd.scalar_type();
    npu_preparation::check_tensor({input}, out, output_dtype_0, output_size_0);
    npu_preparation::check_tensor({input}, save_mean, output_dtype_1, output_size_1);
    npu_preparation::check_tensor({input}, save_invstd, output_dtype_2, output_size_2);
    EXEC_NPU_CMD(aclnnBatchNorm, input, weight, bias, running_mean, running_var, training, momentum, eps, out, save_mean, save_invstd);
    return std::forward_as_tuple(out, save_mean, save_invstd);
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> native_group_norm(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t N, int64_t C, int64_t HxW, int64_t group, double eps)
{
    DO_COMPATIBILITY(aclnnGroupNorm, acl_op::native_group_norm(input, weight, bias, N, C, HxW, group, eps));
    auto output_size_0 = input.sizes();
    auto output_size_1 = {N, group};
    auto output_dtype_0 = input.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                input.options().dtype(output_dtype_0));
    at::Tensor out2 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                input.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGroupNorm, input, weight, bias, N, C, HxW, group, eps, out0, out1, out2);
    return std::make_tuple(std::move(out0), std::move(out1), std::move(out2));
}

at::Tensor neg(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnNeg, acl_op::neg(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnNeg, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & neg_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnNeg, acl_op::neg_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnNeg, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & neg_(at::Tensor & self)
{
    return op_api::neg_out(self, self);
}

at::Tensor pow(const at::Scalar & self, const at::Tensor & exponent)
{
    DO_COMPATIBILITY(aclnnPowScalarTensor, acl_op::pow(self, exponent));
    auto output_size_0 = exponent.sizes();
    auto output_dtype_0 = at::result_type(self, exponent);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                exponent.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnPowScalarTensor, self, exponent, out);
    at::namedinference::propagate_names(out, exponent);
    return out;
}

at::Tensor & pow_out(const at::Scalar & self, const at::Tensor & exponent, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnPowScalarTensor, acl_op::pow_out(self, exponent, out));
    auto output_size_0 = exponent.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({exponent}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnPowScalarTensor, self, exponent, out);
    at::namedinference::propagate_names(out, exponent);
    return out;
}

at::Tensor pow(const at::Tensor & self, const at::Scalar & exponent)
{
    DO_COMPATIBILITY(aclnnPowTensorScalar, acl_op::pow(self, exponent));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = at::result_type(self, exponent);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnPowTensorScalar, self, exponent, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & pow_out(const at::Tensor & self, const at::Scalar & exponent, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnPowTensorScalar, acl_op::pow_out(self, exponent, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = at::result_type(self, exponent);
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnPowTensorScalar, self, exponent, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & pow_(at::Tensor & self, const at::Scalar & exponent)
{
    DO_COMPATIBILITY(aclnnInplacePowTensorScalar, acl_op::pow_(self, exponent));
    EXEC_NPU_CMD(aclnnInplacePowTensorScalar, self, exponent);
    return self;
}

at::Tensor polar(const at::Tensor & abs, const at::Tensor & angle)
{
    DO_COMPATIBILITY(aclnnPolar, acl_op::polar(abs, angle));
    auto output_size_0 = op_infer::broadcast_ops_npu_output_size(abs, angle);
    auto output_dtype_0 = polar_out_dtype(abs, angle);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                abs.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnPolar, abs, angle, out);
    return out;
}

at::Tensor & polar_out(const at::Tensor & abs, const at::Tensor & angle, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnPolar, acl_op::polar_out(abs, angle, out));
    auto output_size_0 = op_infer::broadcast_ops_npu_output_size(abs, angle);
    auto output_dtype_0 = polar_out_dtype(abs, angle);
    npu_preparation::check_tensor({abs, angle}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnPolar, abs, angle, out);
    return out;
}

at::Tensor reciprocal(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnReciprocal, acl_op::reciprocal(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReciprocal, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & reciprocal_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnReciprocal, acl_op::reciprocal_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReciprocal, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & reciprocal_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceReciprocal, acl_op::reciprocal_(self));
    EXEC_NPU_CMD(aclnnInplaceReciprocal, self);
    return self;
}

at::Tensor reflection_pad1d(const at::Tensor & self, at::IntArrayRef padding)
{
    DO_COMPATIBILITY(aclnnReflectionPad1d, acl_op::reflection_pad1d(self, padding));
    auto output_size_0 = reflection_pad1d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReflectionPad1d, self, padding, out);
    return out;
}

at::Tensor & reflection_pad1d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnReflectionPad1d, acl_op::reflection_pad1d_out(self, padding, out));
    auto output_size_0 = reflection_pad1d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReflectionPad1d, self, padding, out);
    return out;
}

at::Tensor reflection_pad1d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding)
{
    DO_COMPATIBILITY(aclnnReflectionPad1dBackward, acl_op::reflection_pad1d_backward(grad_output, self, padding));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReflectionPad1dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor & reflection_pad1d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnReflectionPad1dBackward, acl_op::reflection_pad1d_backward_out(grad_output, self, padding, grad_input));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReflectionPad1dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor reflection_pad2d(const at::Tensor & self, at::IntArrayRef padding)
{
    DO_COMPATIBILITY(aclnnReflectionPad2d, acl_op::reflection_pad2d(self, padding));
    auto output_size_0 = reflection_pad2d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReflectionPad2d, self, padding, out);
    return out;
}

at::Tensor & reflection_pad2d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnReflectionPad2d, acl_op::reflection_pad2d_out(self, padding, out));
    auto output_size_0 = reflection_pad2d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReflectionPad2d, self, padding, out);
    return out;
}

at::Tensor reflection_pad2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding)
{
    DO_COMPATIBILITY(aclnnReflectionPad2dBackward, acl_op::reflection_pad2d_backward(grad_output, self, padding));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReflectionPad2dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor & reflection_pad2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnReflectionPad2dBackward, acl_op::reflection_pad2d_backward_out(grad_output, self, padding, grad_input));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReflectionPad2dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor reflection_pad3d(const at::Tensor & self, at::IntArrayRef padding)
{
    auto output_size_0 = reflection_pad3d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReflectionPad3d, self, padding, out);
    return out;
}

at::Tensor & reflection_pad3d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out)
{
    auto output_size_0 = reflection_pad3d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReflectionPad3d, self, padding, out);
    return out;
}

at::Tensor reflection_pad3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReflectionPad3dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor & reflection_pad3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReflectionPad3dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor relu(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnRelu, acl_op::relu(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnRelu, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & relu_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceRelu, acl_op::relu_(self));
    EXEC_NPU_CMD(aclnnInplaceRelu, self);
    return self;
}

at::Tensor repeat(const at::Tensor & self, at::IntArrayRef repeats)
{
    DO_COMPATIBILITY(aclnnRepeat, acl_op::repeat(self, repeats));
    auto output_size_0 = repeat_npu_output_size(self, repeats);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnRepeat, self, repeats, out);
    return out;
}

at::Tensor replication_pad1d(const at::Tensor & self, at::IntArrayRef padding)
{
    DO_COMPATIBILITY(aclnnReplicationPad1d, acl_op::replication_pad1d(self, padding));
    auto output_size_0 = replication_pad1d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReplicationPad1d, self, padding, out);
    return out;
}

at::Tensor & replication_pad1d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnReplicationPad1d, acl_op::replication_pad1d_out(self, padding, out));
    auto output_size_0 = replication_pad1d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReplicationPad1d, self, padding, out);
    return out;
}

at::Tensor replication_pad1d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding)
{
    DO_COMPATIBILITY(aclnnReplicationPad1dBackward, acl_op::replication_pad1d_backward(grad_output, self, padding));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReplicationPad1dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor & replication_pad1d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnReplicationPad1dBackward, acl_op::replication_pad1d_backward_out(grad_output, self, padding, grad_input));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReplicationPad1dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor replication_pad2d(const at::Tensor & self, at::IntArrayRef padding)
{
    DO_COMPATIBILITY(aclnnReplicationPad2d, acl_op::replication_pad2d(self, padding));
    auto output_size_0 = replication_pad2d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReplicationPad2d, self, padding, out);
    return out;
}

at::Tensor & replication_pad2d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnReplicationPad2d, acl_op::replication_pad2d_out(self, padding, out));
    auto output_size_0 = replication_pad2d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReplicationPad2d, self, padding, out);
    return out;
}

at::Tensor replication_pad2d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding)
{
    DO_COMPATIBILITY(aclnnReplicationPad2dBackward, acl_op::replication_pad2d_backward(grad_output, self, padding));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReplicationPad2dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor & replication_pad2d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnReplicationPad2dBackward, acl_op::replication_pad2d_backward_out(grad_output, self, padding, grad_input));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReplicationPad2dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor replication_pad3d(const at::Tensor & self, at::IntArrayRef padding)
{
    auto output_size_0 = replication_pad3d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReplicationPad3d, self, padding, out);
    return out;
}

at::Tensor & replication_pad3d_out(const at::Tensor & self, at::IntArrayRef padding, at::Tensor & out)
{
    auto output_size_0 = replication_pad3d_npu_out_size(self, padding);
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReplicationPad3d, self, padding, out);
    return out;
}

at::Tensor replication_pad3d_backward(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnReplicationPad3dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor & replication_pad3d_backward_out(const at::Tensor & grad_output, const at::Tensor & self, at::IntArrayRef padding, at::Tensor & grad_input)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnReplicationPad3dBackward, grad_output, self, padding, grad_input);
    return grad_input;
}

at::Tensor roll(const at::Tensor & self, at::IntArrayRef shifts, at::IntArrayRef dims)
{
    DO_COMPATIBILITY(aclnnRoll, acl_op::roll(self, shifts, dims));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnRoll, self, shifts, dims, grad_input);
    return grad_input;
}

at::Tensor round(const at::Tensor & self)
{
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
        return acl_op::round(self);
    }
    DO_COMPATIBILITY(aclnnRound, acl_op::round(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnRound, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & round_out(const at::Tensor & self, at::Tensor & out)
{
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
        return acl_op::round_out(self, out);
    }
    DO_COMPATIBILITY(aclnnRound, acl_op::round_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnRound, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & round_(at::Tensor & self)
{
    return op_api::round_out(self, self);
}

at::Tensor rsqrt(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnRsqrt, acl_op::rsqrt(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = isIntegralType(self.scalar_type(), true) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnRsqrt, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & rsqrt_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnRsqrt, acl_op::rsqrt_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnRsqrt, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & rsqrt_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceRsqrt, acl_op::rsqrt_(self));
    EXEC_NPU_CMD(aclnnInplaceRsqrt, self);
    return self;
}

at::Tensor rsub(const at::Tensor & self, const at::Scalar & other, const at::Scalar & alpha)
{
    DO_COMPATIBILITY(aclnnRsubs, acl_op::rsub(self, other, alpha));
    at::Tensor self_cp = self;
    if (npu_preparation::IsCPUScalar(self)) {
        at::Scalar self_scalar = self.item();
        self_cp = npu_preparation::copy_scalar_to_device(self_scalar, self.scalar_type(), c10::Device(c10::DeviceType::PrivateUse1));
    }
    auto output_size_0 = self_cp.sizes();
    auto output_dtype_0 = at::native::result_type(self_cp, other);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self_cp.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnRsubs, self_cp, other, alpha, out);
    return out;
}

at::Tensor rsub(const at::Tensor & self, const at::Tensor & other, const at::Scalar & alpha)
{
    DO_COMPATIBILITY(aclnnRsub, acl_op::rsub(self, other, alpha));
    DO_COMPATIBILITY(aclnnRsubs, acl_op::rsub(self, other, alpha));
    at::Tensor self_cp = self;
    if (npu_preparation::IsCPUScalar(self)) {
        at::Scalar self_scalar = self.item();
        self_cp = npu_preparation::copy_scalar_to_device(self_scalar, self.scalar_type(), other.device());
    }
    auto output_size_0 = broadcast_ops_npu_output_size(self_cp, other);
    auto output_dtype_0 = at::native::result_type(self_cp, other);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self_cp.options().dtype(output_dtype_0));
    if (npu_preparation::IsCPUScalar(other)) {
        at::Scalar other_scalar = other.item();
        EXEC_NPU_CMD(aclnnRsubs, self_cp, other_scalar, alpha, out);
        return out;
    }
    EXEC_NPU_CMD(aclnnRsub, self_cp, other, alpha, out);
    return out;
}

at::Tensor searchsorted(const at::Tensor & sorted_sequence, const at::Scalar & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<at::Tensor> & sorter)
{
    DO_COMPATIBILITY(aclnnSearchSorteds, acl_op::searchsorted(sorted_sequence, self, out_int32, right, side, sorter));
    auto right_eff = (op_plugin::searchsorted_validate_scalar_op(sorted_sequence, self, out_int32, right, side, sorter), op_plugin::warn_if_searchsorted_scalar_inputs_noncontiguous(sorted_sequence, sorter), op_plugin::resolve_searchsorted_effective_right(right, side));
    auto output_size_0 = c10::SmallVector<int64_t, op_infer::SIZE>{};
    auto output_dtype_0 = out_int32 ? at::kInt : at::kLong;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                sorted_sequence.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSearchSorteds, sorted_sequence, self, out_int32, right_eff, sorter, out);
    return out;
}

at::Tensor searchsorted(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<at::Tensor> & sorter)
{
    DO_COMPATIBILITY(aclnnSearchSorted, acl_op::searchsorted(sorted_sequence, self, out_int32, right, side, sorter));
    auto right_eff = (op_plugin::searchsorted_validate_core_no_output(sorted_sequence, self, out_int32, right, side, sorter), op_plugin::warn_if_searchsorted_inputs_noncontiguous(sorted_sequence, self, sorter), op_plugin::resolve_searchsorted_effective_right(right, side));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out_int32 ? at::kInt : at::kLong;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                sorted_sequence.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSearchSorted, sorted_sequence, self, out_int32, right_eff, sorter, out);
    return out;
}

at::Tensor & searchsorted_out(const at::Tensor & sorted_sequence, const at::Tensor & self, bool out_int32, bool right, c10::optional<c10::string_view> side, const c10::optional<at::Tensor> & sorter, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSearchSorted, acl_op::searchsorted_out(sorted_sequence, self, out_int32, right, side, sorter, out));
    auto right_eff = (op_plugin::searchsorted_validate_tensor_out_op(sorted_sequence, self, out, out_int32, right, side, sorter), op_plugin::warn_if_searchsorted_inputs_noncontiguous(sorted_sequence, self, sorter), op_plugin::resolve_searchsorted_effective_right(right, side));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out_int32 ? at::kInt : at::kLong;
    npu_preparation::check_tensor({sorted_sequence, self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSearchSorted, sorted_sequence, self, out_int32, right_eff, sorter, out);
    return out;
}

at::Tensor sigmoid(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnSigmoid, acl_op::sigmoid(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = (isIntegralType(self.scalar_type(), true)) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSigmoid, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & sigmoid_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSigmoid, acl_op::sigmoid_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSigmoid, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & sigmoid_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceSigmoid, acl_op::sigmoid_(self));
    EXEC_NPU_CMD(aclnnInplaceSigmoid, self);
    return self;
}

at::Tensor sigmoid_backward(const at::Tensor & grad_output, const at::Tensor & output)
{
    DO_COMPATIBILITY(aclnnSigmoidBackward, acl_op::sigmoid_backward(grad_output, output));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_output, output);
    auto output_dtype_0 = at::native::result_type(grad_output, output);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSigmoidBackward, grad_output, output, grad_input);
    return grad_input;
}

at::Tensor & sigmoid_backward_out(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnSigmoidBackward, acl_op::sigmoid_backward_out(grad_output, output, grad_input));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_output, output);
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, output}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSigmoidBackward, grad_output, output, grad_input);
    return grad_input;
}

at::Tensor sign(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnSign, acl_op::sign(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSign, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & sign_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSign, acl_op::sign_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSign, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & sign_(at::Tensor & self)
{
    return op_api::sign_out(self, self);
}

at::Tensor & signbit_out(const at::Tensor & self, at::Tensor & out)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSignbit, self, out);
    return out;
}

at::Tensor silu(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnSilu, acl_op::silu(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSilu, self, out);
    return out;
}

at::Tensor & silu_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSilu, acl_op::silu_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSilu, self, out);
    return out;
}

at::Tensor & silu_(at::Tensor & self)
{
    return op_api::silu_out(self, self);
}

at::Tensor sin(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnSin, acl_op::sin(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = (isIntegralType(self.scalar_type(), true)) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSin, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & sin_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSin, acl_op::sin_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSin, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & sin_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceSin, acl_op::sin_(self));
    EXEC_NPU_CMD(aclnnInplaceSin, self);
    return self;
}

at::Tensor sinc(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnSinc, acl_op::sinc(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = (isIntegralType(self.scalar_type(), true)) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSinc, self, out);
    return out;
}

at::Tensor & sinc_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSinc, acl_op::sinc_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSinc, self, out);
    return out;
}

at::Tensor & sinc_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceSinc, acl_op::sinc_(self));
    EXEC_NPU_CMD(aclnnInplaceSinc, self);
    return self;
}

at::Tensor sinh(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnSinh, acl_op::sinh(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = (isIntegralType(self.scalar_type(), true)) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSinh, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & sinh_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSinh, acl_op::sinh_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSinh, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & sinh_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceSinh, acl_op::sinh_(self));
    EXEC_NPU_CMD(aclnnInplaceSinh, self);
    return self;
}

at::Tensor soft_margin_loss(const at::Tensor & self, const at::Tensor & target, int64_t reduction)
{
    DO_COMPATIBILITY(aclnnSoftMarginLoss, acl_op::soft_margin_loss(self, target, reduction));
    auto output_size_0 = (reduction == at::Reduction::None) ? broadcast_ops_npu_output_size(self, target) : c10::SmallVector<int64_t, op_infer::SIZE>{};
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSoftMarginLoss, self, target, reduction, out);
    return out;
}

at::Tensor & soft_margin_loss_out(const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSoftMarginLoss, acl_op::soft_margin_loss_out(self, target, reduction, out));
    auto output_size_0 = (reduction == at::Reduction::None) ? broadcast_ops_npu_output_size(self, target) : c10::SmallVector<int64_t, op_infer::SIZE>{};
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, target}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSoftMarginLoss, self, target, reduction, out);
    return out;
}

at::Tensor soft_margin_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction)
{
    DO_COMPATIBILITY(aclnnSoftMarginLossBackward, acl_op::soft_margin_loss_backward(grad_output, self, target, reduction));
    auto output_size_0 = broadcast_ops_npu_output_size(broadcast_ops_npu_output_size(grad_output.sizes(), self.sizes()), target.sizes());
    auto output_dtype_0 = self.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSoftMarginLossBackward, grad_output, self, target, reduction, grad_input);
    return grad_input;
}

at::Tensor & soft_margin_loss_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnSoftMarginLossBackward, acl_op::soft_margin_loss_backward_out(grad_output, self, target, reduction, grad_input));
    auto output_size_0 = broadcast_ops_npu_output_size(broadcast_ops_npu_output_size(grad_output.sizes(), self.sizes()), target.sizes());
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self, target}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSoftMarginLossBackward, grad_output, self, target, reduction, grad_input);
    return grad_input;
}

at::Tensor softplus(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold)
{
    DO_COMPATIBILITY(aclnnSoftplus, acl_op::softplus(self, beta, threshold));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSoftplus, self, beta, threshold, out);
    return out;
}

at::Tensor & softplus_out(const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSoftplus, acl_op::softplus_out(self, beta, threshold, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSoftplus, self, beta, threshold, out);
    return out;
}

at::Tensor & softplus_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & beta, const at::Scalar & threshold, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnSoftplusBackward, acl_op::softplus_backward_out(grad_output, self, beta, threshold, grad_input));
    auto output_size_0 = grad_input.sizes();
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSoftplusBackward, grad_output, self, beta, threshold, grad_input);
    return grad_input;
}

at::Tensor softshrink(const at::Tensor & self, const at::Scalar & lambd)
{
    DO_COMPATIBILITY(aclnnSoftshrink, acl_op::softshrink(self, lambd));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSoftshrink, self, lambd, out);
    return out;
}

at::Tensor & softshrink_out(const at::Tensor & self, const at::Scalar & lambd, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSoftshrink, acl_op::softshrink_out(self, lambd, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSoftshrink, self, lambd, out);
    return out;
}

at::Tensor softshrink_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd)
{
    DO_COMPATIBILITY(aclnnSoftshrinkBackward, acl_op::softshrink_backward(grad_output, self, lambd));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_output, self);
    auto output_dtype_0 = at::native::result_type(grad_output, self);
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSoftshrinkBackward, grad_output, self, lambd, grad_input);
    return grad_input;
}

at::Tensor & softshrink_backward_out(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & lambd, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnSoftshrinkBackward, acl_op::softshrink_backward_out(grad_output, self, lambd, grad_input));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_output, self);
    auto output_dtype_0 = grad_input.scalar_type();
    npu_preparation::check_tensor({grad_output, self}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSoftshrinkBackward, grad_output, self, lambd, grad_input);
    return grad_input;
}

at::Tensor sqrt(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnSqrt, acl_op::sqrt(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = (isIntegralType(self.scalar_type(), true)) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSqrt, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & sqrt_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnSqrt, acl_op::sqrt_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnSqrt, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & sqrt_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceSqrt, acl_op::sqrt_(self));
    EXEC_NPU_CMD(aclnnInplaceSqrt, self);
    return self;
}

at::Tensor stack(at::TensorList tensors, int64_t dim)
{
    DO_COMPATIBILITY(aclnnStack, acl_op::stack(tensors, dim));
    auto output_size_0 = stack_output_size(tensors, dim);
    auto output_dtype_0 = at::native::result_type(tensors);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                tensors[0].options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnStack, tensors, dim, out);
    return out;
}

at::Tensor & stack_out(at::TensorList tensors, int64_t dim, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnStack, acl_op::stack_out(tensors, dim, out));
    auto output_size_0 = stack_output_size(tensors, dim);
    auto output_dtype_0 = tensors[0].scalar_type();
    npu_preparation::check_tensor({tensors[0]}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnStack, tensors, dim, out);
    return out;
}

at::Tensor take(const at::Tensor & self, const at::Tensor & index)
{
    DO_COMPATIBILITY(aclnnTake, acl_op::take(self, index));
    auto output_size_0 = index.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnTake, self, index, out);
    return out;
}

at::Tensor & take_out(const at::Tensor & self, const at::Tensor & index, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnTake, acl_op::take_out(self, index, out));
    auto output_size_0 = index.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self, index}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnTake, self, index, out);
    return out;
}

at::Tensor tan(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnTan, acl_op::tan(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = (isIntegralType(self.scalar_type(), true)) ? at::kFloat : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnTan, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & tan_out(const at::Tensor & self, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnTan, acl_op::tan_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnTan, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & tan_(at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnInplaceTan, acl_op::tan_(self));
    EXEC_NPU_CMD(aclnnInplaceTan, self);
    return self;
}

at::Tensor tanh_backward(const at::Tensor & grad_output, const at::Tensor & output)
{
    DO_COMPATIBILITY(aclnnTanhBackward, acl_op::tanh_backward(grad_output, output));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_output, output);
    auto output_dtype_0 = grad_output.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnTanhBackward, grad_output, output, grad_input);
    return grad_input;
}

at::Tensor & tanh_backward_out(const at::Tensor & grad_output, const at::Tensor & output, at::Tensor & grad_input)
{
    DO_COMPATIBILITY(aclnnTanhBackward, acl_op::tanh_backward_out(grad_output, output, grad_input));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_output, output);
    auto output_dtype_0 = grad_output.scalar_type();
    npu_preparation::check_tensor({grad_output, output}, grad_input, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnTanhBackward, grad_output, output, grad_input);
    return grad_input;
}

at::Tensor threshold_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Scalar & threshold)
{
    DO_COMPATIBILITY(aclnnThresholdBackward, acl_op::threshold_backward(grad_output, self, threshold));
    auto output_size_0 = broadcast_ops_npu_output_size(grad_output, self);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnThresholdBackward, grad_output, self, threshold, out);
    return out;
}

at::Tensor trace(const at::Tensor & self)
{
    DO_COMPATIBILITY(aclnnTrace, acl_op::trace(self));
    auto output_size_0 = c10::SmallVector<int64_t, op_infer::N>{};
    auto output_dtype_0 = (isIntegralType(self.scalar_type(), true)) ? at::kLong : self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnTrace, self, out);
    return out;
}

::std::tuple<at::Tensor &,at::Tensor &> triangular_solve_out(const at::Tensor & self, const at::Tensor & A, bool upper, bool transpose, bool unitriangular, at::Tensor & X, at::Tensor & M)
{
    DO_COMPATIBILITY(aclnnTriangularSolve, acl_op::triangular_solve_out(self, A, upper, transpose, unitriangular, X, M));
    auto output_sizes = op_infer::triangular_solve_output_size(self, A);
    auto output_size_0 = std::get<0>(output_sizes);
    auto output_size_1 = std::get<1>(output_sizes);
    auto output_dtype_0 = X.scalar_type();
    auto output_dtype_1 = M.scalar_type();
    npu_preparation::check_tensor({self, A}, X, output_dtype_0, output_size_0);
    npu_preparation::check_tensor({self, A}, M, output_dtype_1, output_size_1);
    EXEC_NPU_CMD(aclnnTriangularSolve, self, A, upper, transpose, unitriangular, X, M);
    return std::forward_as_tuple(X, M);
}

at::Tensor tril(const at::Tensor & self, int64_t diagonal)
{
    DO_COMPATIBILITY(aclnnTril, acl_op::tril(self, diagonal));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnTril, self, diagonal, out);
    return out;
}

at::Tensor & tril_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnTril, acl_op::tril_out(self, diagonal, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnTril, self, diagonal, out);
    return out;
}

at::Tensor & tril_(at::Tensor & self, int64_t diagonal)
{
    DO_COMPATIBILITY(aclnnInplaceTril, acl_op::tril_(self, diagonal));
    EXEC_NPU_CMD(aclnnInplaceTril, self, diagonal);
    return self;
}

at::Tensor triu(const at::Tensor & self, int64_t diagonal)
{
    DO_COMPATIBILITY(aclnnTriu, acl_op::triu(self, diagonal));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnTriu, self, diagonal, out);
    return out;
}

at::Tensor & triu_out(const at::Tensor & self, int64_t diagonal, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnTriu, acl_op::triu_out(self, diagonal, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnTriu, self, diagonal, out);
    return out;
}

at::Tensor & triu_(at::Tensor & self, int64_t diagonal)
{
    DO_COMPATIBILITY(aclnnInplaceTriu, acl_op::triu_(self, diagonal));
    EXEC_NPU_CMD(aclnnInplaceTriu, self, diagonal);
    return self;
}

at::Tensor trunc(const at::Tensor & self)
{
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
        return acl_op::trunc(self);
    }
    DO_COMPATIBILITY(aclnnTrunc, acl_op::trunc(self));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnTrunc, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & trunc_out(const at::Tensor & self, at::Tensor & out)
{
    if (at::isIntegralType(self.scalar_type(), /*includeBool=*/false)) {
        return acl_op::trunc_out(self, out);
    }
    DO_COMPATIBILITY(aclnnTrunc, acl_op::trunc_out(self, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnTrunc, self, out);
    at::namedinference::propagate_names(out, self);
    return out;
}

at::Tensor & trunc_(at::Tensor & self)
{
    return op_api::trunc_out(self, self);
}

at::Tensor vdot(const at::Tensor & self, const at::Tensor & other)
{
    DO_COMPATIBILITY(aclnnDot, acl_op::vdot(self, other));
    auto output_size_0 = c10::SmallVector<int64_t, op_infer::N>{};
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnDot, self, other, out);
    return out;
}

at::Tensor & vdot_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnDot, acl_op::vdot_out(self, other, out));
    auto output_size_0 = c10::SmallVector<int64_t, op_infer::N>{};
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, other}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnDot, self, other, out);
    return out;
}

at::Tensor & xlogy_out(const at::Tensor & self, const at::Scalar & other, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnXLogYScalarOther, acl_op::xlogy_out(self, other, out));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnXLogYScalarOther, self, other, out);
    return out;
}

at::Tensor & xlogy_out(const at::Scalar & self, const at::Tensor & other, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnXLogYScalarSelf, acl_op::xlogy_out(self, other, out));
    auto output_size_0 = other.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({other}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnXLogYScalarSelf, self, other, out);
    return out;
}

at::Tensor & xlogy_out(const at::Tensor & self, const at::Tensor & other, at::Tensor & out)
{
    DO_COMPATIBILITY(aclnnXLogYTensor, acl_op::xlogy_out(self, other, out));
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({self, other}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnXLogYTensor, self, other, out);
    return out;
}

at::Tensor xlogy(const at::Tensor & self, const at::Scalar & other)
{
    DO_COMPATIBILITY(aclnnXLogYScalarOther, acl_op::xlogy(self, other));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = (isIntegralType(at::result_type(self, other), true)) ? at::kFloat : at::result_type(self, other);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnXLogYScalarOther, self, other, out);
    return out;
}

at::Tensor xlogy(const at::Scalar & self, const at::Tensor & other)
{
    DO_COMPATIBILITY(aclnnXLogYScalarSelf, acl_op::xlogy(self, other));
    auto output_size_0 = other.sizes();
    auto output_dtype_0 = (isIntegralType(at::result_type(self, other), true)) ? at::kFloat : at::result_type(self, other);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                other.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnXLogYScalarSelf, self, other, out);
    return out;
}

at::Tensor xlogy(const at::Tensor & self, const at::Tensor & other)
{
    DO_COMPATIBILITY(aclnnXLogYTensor, acl_op::xlogy(self, other));
    auto output_size_0 = broadcast_ops_npu_output_size(self, other);
    auto output_dtype_0 = (isIntegralType(at::result_type(self, other), true)) ? at::kFloat : at::result_type(self, other);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnXLogYTensor, self, other, out);
    return out;
}

at::Tensor & xlogy_(at::Tensor & self, const at::Scalar & other)
{
    DO_COMPATIBILITY(aclnnInplaceXLogYScalarOther, acl_op::xlogy_(self, other));
    EXEC_NPU_CMD(aclnnInplaceXLogYScalarOther, self, other);
    return self;
}

at::Tensor & xlogy_(at::Tensor & self, const at::Tensor & other)
{
    return op_api::xlogy_out(self, other, self);
}

at::Tensor & ffn_worker_scheduler_(at::Tensor & self, int64_t sync_group_size, int64_t execute_mode)
{
    EXEC_NPU_CMD(aclnnInplaceFfnWorkerScheduler, self, sync_group_size, execute_mode);
    return self;
}

at::Tensor & attention_worker_scheduler_(at::Tensor & self)
{
    EXEC_NPU_CMD(aclnnInplaceAttentionWorkerScheduler, self);
    return self;
}

at::Tensor npu_gelu(const at::Tensor & self, c10::string_view approximate)
{
    auto approximate_mode = npu_gelu_approximate_mode(approximate);
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGeluV2, self, approximate_mode, out);
    return out;
}

at::Tensor npu_gelu_backward(const at::Tensor & grad_output, const at::Tensor & self, c10::string_view approximate)
{
    auto approximate_str = npu_gelu_approximate_str(approximate);
    auto approximate_ptr = const_cast<char *>(approximate_str.c_str());
    auto output_size_0 = broadcast_ops_npu_output_size(grad_output, self);
    auto output_dtype_0 = at::native::result_type(grad_output, self);
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGeluBackwardV2, grad_output, self, approximate_ptr, out);
    return out;
}

at::Tensor npu_gelu_mul(const at::Tensor & input, c10::string_view approximate)
{
    auto approximate_str = npu_gelu_approximate_str(approximate);
    auto approximate_ptr = const_cast<char*>(approximate_str.c_str());
    auto output_size_0 = npu_gelu_mul_output_size(input);
    auto output_dtype_0 = input.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGeluMul, input, approximate_ptr, out);
    return out;
}

at::Tensor kl_div_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target)
{
    DO_COMPATIBILITY(aclnnKlDivBackward, acl_op::kl_div_backward(grad_output, self, target, reduction, log_target));
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnKlDivBackward, grad_output, self, target, reduction, log_target, out);
    return out;
}

at::Tensor kl_div_target_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction, bool log_target)
{
    auto output_size_0 = target.sizes();
    auto output_dtype_0 = target.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnKlDivTargetBackward, grad_output, self, target, reduction, log_target, out);
    return out;
}

at::Tensor l1_loss_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, int64_t reduction)
{
    DO_COMPATIBILITY(aclnnL1LossBackward, acl_op::l1_loss_backward(grad_output, self, target, reduction));
    auto output_size_0 = broadcast_ops_npu_output_size(broadcast_ops_npu_output_size(self, target), grad_output.sizes());
    auto output_dtype_0 = promoteTypes(grad_output.scalar_type(), at::native::result_type(target, self));
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnL1LossBackward, grad_output, self, target, reduction, grad_input);
    return grad_input;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_add_rms_norm(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, double epsilon)
{
    DO_COMPATIBILITY(aclnnAddRmsNorm, acl_op::npu_add_rms_norm(x1, x2, gamma, epsilon));
    auto output_size_0 = rms_norm_npu_output_size(x1, gamma)[0];
    auto output_size_1 = rms_norm_npu_output_size(x1, gamma)[1];
    auto output_dtype_0 = x1.scalar_type();
    auto output_dtype_1 = at::kFloat;
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x1.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                x1.options().dtype(output_dtype_1));
    at::Tensor out2 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x1.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnAddRmsNorm, x1, x2, gamma, epsilon, out0, out1, out2);
    return std::make_tuple(std::move(out0), std::move(out1), std::move(out2));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_rms_norm_cast(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, double epsilon)
{
    auto output_size_0 = rms_norm_npu_output_size(x1, gamma)[0];
    auto output_size_1 = rms_norm_npu_output_size(x1, gamma)[1];
    auto output_dtype_0 = at::kFloat;
    auto output_dtype_1 = x1.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x1.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x1.options().dtype(output_dtype_1));
    at::Tensor out2 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                x1.options().dtype(output_dtype_0));
    at::Tensor out3 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x1.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnAddRmsNormCast, x1, x2, gamma, epsilon, out0, out1, out2, out3);
    return std::make_tuple(std::move(out0), std::move(out1), std::move(out2), std::move(out3));
}

at::Tensor npu_binary_cross_entropy_with_logits_backward(const at::Tensor & grad_output, const at::Tensor & self, const at::Tensor & target, const c10::optional<at::Tensor> & weight_opt, const c10::optional<at::Tensor> & pos_weight_opt, int64_t reduction)
{
    DO_COMPATIBILITY(aclnnBinaryCrossEntropyWithLogitsBackward, acl_op::npu_binary_cross_entropy_with_logits_backward(grad_output, self, target, weight_opt, pos_weight_opt, reduction));
    auto output_size_0 = target.sizes();
    auto output_dtype_0 = target.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnBinaryCrossEntropyWithLogitsBackward, grad_output, self, target, weight_opt, pos_weight_opt, reduction, out);
    return out;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_fused_floyd_attention(const at::Tensor & query_ik, const at::Tensor & key_ij, const at::Tensor & value_ij, const at::Tensor & key_jk, const at::Tensor & value_jk, const c10::optional<at::Tensor> & atten_mask, double scale_value)
{
    auto output_size_0 = {query_ik.size(0), query_ik.size(1), query_ik.size(2), query_ik.size(3), 8L};
    auto output_size_1 = query_ik.sizes();
    auto output_dtype_0 = at::kFloat;
    auto output_dtype_1 = query_ik.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                query_ik.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                query_ik.options().dtype(output_dtype_0));
    at::Tensor out2 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                query_ik.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnFusedFloydAttention, query_ik, key_ij, value_ij, key_jk, value_jk, atten_mask, scale_value, out0, out1, out2);
    return std::make_tuple(std::move(out0), std::move(out1), std::move(out2));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_fused_floyd_attention_backward(const at::Tensor & grad_output, const at::Tensor & query_ik, const at::Tensor & key_ij, const at::Tensor & value_ij, const at::Tensor & key_jk, const at::Tensor & value_jk, const at::Tensor & attention_out, const at::Tensor & softmax_max, const at::Tensor & softmax_sum, const c10::optional<at::Tensor> & atten_mask, double scale_value)
{
    auto output_size_0 = query_ik.sizes();
    auto output_size_1 = key_ij.sizes();
    auto output_size_2 = value_ij.sizes();
    auto output_size_3 = key_jk.sizes();
    auto output_size_4 = value_jk.sizes();
    auto output_dtype_0 = query_ik.scalar_type();
    auto output_dtype_1 = key_ij.scalar_type();
    auto output_dtype_2 = value_ij.scalar_type();
    auto output_dtype_3 = key_jk.scalar_type();
    auto output_dtype_4 = value_jk.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                grad_output.options().dtype(output_dtype_1));
    at::Tensor out2 = npu_preparation::apply_tensor_without_format(output_size_2,
                                                                grad_output.options().dtype(output_dtype_2));
    at::Tensor out3 = npu_preparation::apply_tensor_without_format(output_size_3,
                                                                grad_output.options().dtype(output_dtype_3));
    at::Tensor out4 = npu_preparation::apply_tensor_without_format(output_size_4,
                                                                grad_output.options().dtype(output_dtype_4));
    EXEC_NPU_CMD(aclnnFusedFloydAttentionGrad, query_ik, key_ij, value_ij, key_jk, value_jk, grad_output, atten_mask, softmax_max, softmax_sum, attention_out, scale_value, out0, out1, out2, out3, out4);
    return std::make_tuple(std::move(out0), std::move(out1), std::move(out2), std::move(out3), std::move(out4));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_silu(const at::Tensor & input, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & bias, int64_t group, double eps)
{
    auto output_size_0 = input.sizes();
    auto output_size_1 = {input.size(0), group};
    auto output_dtype_0 = input.scalar_type();
    auto output_dtype_1 = npu_group_norm_silu_dst_type(input, weight, bias);
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                input.options().dtype(output_dtype_1));
    at::Tensor out2 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                input.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnGroupNormSilu, input, weight, bias, group, eps, out0, out1, out2);
    return std::make_tuple(std::move(out0), std::move(out1), std::move(out2));
}

::std::tuple<at::Tensor &,at::Tensor &,at::Tensor &,at::Tensor &> npu_hans_encode_out(const at::Tensor & input, bool statistic, bool reshuff, at::Tensor & pdf, at::Tensor & mantissa, at::Tensor & fixed, at::Tensor & var)
{
    auto output_size_0 = pdf.sizes();
    auto output_size_1 = mantissa.sizes();
    auto output_size_2 = fixed.sizes();
    auto output_size_3 = var.sizes();
    auto output_dtype_0 = pdf.scalar_type();
    auto output_dtype_1 = mantissa.scalar_type();
    auto output_dtype_2 = fixed.scalar_type();
    auto output_dtype_3 = var.scalar_type();
    npu_preparation::check_tensor({input}, pdf, output_dtype_0, output_size_0);
    npu_preparation::check_tensor({input}, mantissa, output_dtype_1, output_size_1);
    npu_preparation::check_tensor({input}, fixed, output_dtype_2, output_size_2);
    npu_preparation::check_tensor({input}, var, output_dtype_3, output_size_3);
    EXEC_NPU_CMD(aclnnHansEncode, input, pdf, statistic, reshuff, mantissa, fixed, var);
    return std::forward_as_tuple(pdf, mantissa, fixed, var);
}

at::Tensor & npu_hans_decode_out(const at::Tensor & mantissa, const at::Tensor & fixed, const at::Tensor & var, const at::Tensor & pdf, bool reshuff, at::Tensor & out)
{
    auto output_size_0 = out.sizes();
    auto output_dtype_0 = out.scalar_type();
    npu_preparation::check_tensor({mantissa, fixed, var, pdf}, out, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnHansDecode, mantissa, fixed, var, pdf, reshuff, out);
    return out;
}

at::Tensor npu_masked_softmax_with_rel_pos_bias(const at::Tensor & x, const c10::optional<at::Tensor> & atten_mask, const at::Tensor & relative_pos_bias, double scale_value, int64_t inner_precision_mode)
{
    auto output_size_0 = x.sizes();
    auto output_dtype_0 = x.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMaskedSoftmaxWithRelPosBias, x, atten_mask, relative_pos_bias, scale_value, inner_precision_mode, out);
    return out;
}

at::Tensor npu_mhc_post(const at::Tensor & x, const at::Tensor & h_res, const at::Tensor & h_out, const at::Tensor & h_post)
{
    auto output_size_0 = x.sizes();
    auto output_dtype_0 = x.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMhcPost, x, h_res, h_out, h_post, out);
    return out;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_mhc_post_backward(const at::Tensor & grad_output, const at::Tensor & x, const at::Tensor & h_res, const at::Tensor & h_out, const at::Tensor & h_post)
{
    auto output_size_0 = x.sizes();
    auto output_size_1 = h_res.sizes();
    auto output_size_2 = h_out.sizes();
    auto output_size_3 = h_post.sizes();
    auto output_dtype_0 = x.scalar_type();
    auto output_dtype_1 = h_res.scalar_type();
    auto output_dtype_2 = h_out.scalar_type();
    auto output_dtype_3 = h_post.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    at::Tensor out1 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                grad_output.options().dtype(output_dtype_1));
    at::Tensor out2 = npu_preparation::apply_tensor_without_format(output_size_2,
                                                                grad_output.options().dtype(output_dtype_2));
    at::Tensor out3 = npu_preparation::apply_tensor_without_format(output_size_3,
                                                                grad_output.options().dtype(output_dtype_3));
    EXEC_NPU_CMD(aclnnMhcPostBackward, grad_output, x, h_res, h_out, h_post, out0, out1, out2, out3);
    return std::make_tuple(std::move(out0), std::move(out1), std::move(out2), std::move(out3));
}

at::Tensor npu_moe_compute_expert_tokens(const at::Tensor & sorted_expert_for_source_row, int64_t num_expert)
{
    auto output_size_0 = {num_expert};
    auto output_dtype_0 = sorted_expert_for_source_row.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                sorted_expert_for_source_row.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMoeComputeExpertTokens, sorted_expert_for_source_row, num_expert, out);
    return out;
}

at::Tensor npu_moe_gating_top_k_backward(const at::Tensor & x_norm, const at::Tensor & grad_y, const at::Tensor & expertIdx, int64_t renorm, int64_t norm_type, double routed_scaling_factor, double eps)
{
    auto output_size_0 = x_norm.sizes();
    auto output_dtype_0 = grad_y.scalar_type();
    at::Tensor out0 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x_norm.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMoeGatingTopKBackward, x_norm, grad_y, expertIdx, renorm, norm_type, routed_scaling_factor, eps, out0);
    return out0;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_moe_distribute_dispatch_setup(const at::Tensor & x, const at::Tensor & expert_ids, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const c10::optional<at::Tensor> & scales, const c10::optional<at::Tensor> & x_active_mask, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs, int64_t comm_type, c10::string_view comm_alg, c10::optional<int64_t> y_dtype)
{
    auto group_ep_ptr = const_cast<char *>(group_ep.data());
    auto comm_alg_ptr = const_cast<char *>(comm_alg.data());
    auto out_sizes = npu_moe_distribute_dispatch_setup_out_size(x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, y_dtype);
    auto out_y_dtype = (!y_dtype.has_value()) ? x.scalar_type() : npu_moe_distribute_dispatch_setup_out_dtype(y_dtype);
    auto output_size_0 = out_sizes[0];
    auto output_size_1 = out_sizes[1];
    auto output_size_2 = out_sizes[2];
    auto output_dtype_0 = out_y_dtype;
    auto output_dtype_1 = at::kInt;
    at::Tensor out_y = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x.options().dtype(output_dtype_0));
    at::Tensor out_expand_idx = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                x.options().dtype(output_dtype_1));
    at::Tensor out_comm_cmd_info = npu_preparation::apply_tensor_without_format(output_size_2,
                                                                x.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnMoeDistributeDispatchSetup, x, expert_ids, scales, x_active_mask, group_ep_ptr, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, comm_type, comm_alg_ptr, out_y, out_expand_idx, out_comm_cmd_info);
    return std::make_tuple(std::move(out_y), std::move(out_expand_idx), std::move(out_comm_cmd_info));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_moe_distribute_dispatch_teardown(const at::Tensor & x, const at::Tensor & y, const at::Tensor & expert_ids, const at::Tensor & comm_cmd_info, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t quant_mode, int64_t global_bs, int64_t expert_token_nums_type, int64_t comm_type, c10::string_view comm_alg)
{
    auto out_sizes = npu_moe_distribute_dispatch_teardown_out_size(x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type, shared_expert_rank_num, global_bs, quant_mode);
    auto float8_e8m0_dtype = npu_preparation::convert_to_scalar_type(c10_npu::GetAclDataType(c10_npu::DType::FLOAT8_E8M0));
    auto group_ep_ptr = const_cast<char *>(group_ep.data());
    auto comm_alg_ptr = const_cast<char *>(comm_alg.data());
    auto output_size_0 = out_sizes[0];
    auto output_size_1 = out_sizes[1];
    auto output_size_2 = out_sizes[2];
    auto output_size_3 = out_sizes[3];
    auto output_dtype_0 = y.scalar_type();
    auto output_dtype_1 = (quant_mode == 4) ? float8_e8m0_dtype : at::kFloat;
    auto output_dtype_2 = at::kInt;
    auto output_dtype_3 = at::kLong;
    at::Tensor out_expand_x = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x.options().dtype(output_dtype_0));
    at::Tensor out_dynamic_scales = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                x.options().dtype(output_dtype_1));
    at::Tensor out_assist_info_for_combine = npu_preparation::apply_tensor_without_format(output_size_2,
                                                                x.options().dtype(output_dtype_2));
    at::Tensor out_expert_token_nums = npu_preparation::apply_tensor_without_format(output_size_3,
                                                                x.options().dtype(output_dtype_3));
    EXEC_NPU_CMD(aclnnMoeDistributeDispatchTeardown, x, y, expert_ids, comm_cmd_info, group_ep_ptr, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type, shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type, comm_type, comm_alg_ptr, out_expand_x, out_dynamic_scales, out_assist_info_for_combine, out_expert_token_nums);
    return std::make_tuple(std::move(out_expand_x), std::move(out_dynamic_scales), std::move(out_assist_info_for_combine), std::move(out_expert_token_nums));
}

::std::tuple<at::Tensor,at::Tensor> npu_moe_distribute_combine_setup(const at::Tensor & expand_x, const at::Tensor & expert_ids, const at::Tensor & assist_info_for_combine, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t comm_quant_mode, int64_t comm_type, c10::string_view comm_alg)
{
    auto out_sizes = npu_moe_distribute_combine_setup_out_size(expand_x, ep_world_size);
    auto group_ep_ptr = const_cast<char *>(group_ep.data());
    auto comm_alg_ptr = const_cast<char *>(comm_alg.data());
    auto output_size_0 = out_sizes[0];
    auto output_size_1 = out_sizes[1];
    auto output_dtype_0 = at::kChar;
    auto output_dtype_1 = at::kInt;
    at::Tensor out_quant_expand_x = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                expand_x.options().dtype(output_dtype_0));
    at::Tensor out_comm_cmd_info = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                expand_x.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnMoeDistributeCombineSetup, expand_x, expert_ids, assist_info_for_combine, group_ep_ptr, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, comm_quant_mode, comm_type, comm_alg_ptr, out_quant_expand_x, out_comm_cmd_info);
    return std::make_tuple(std::move(out_quant_expand_x), std::move(out_comm_cmd_info));
}

at::Tensor npu_moe_distribute_combine_teardown(const at::Tensor & expand_x, const at::Tensor & quant_expand_x, const at::Tensor & expert_ids, const at::Tensor & expand_idx, const at::Tensor & expert_scales, const at::Tensor & comm_cmd_info, c10::string_view group_ep, int64_t ep_world_size, int64_t ep_rank_id, int64_t moe_expert_num, const c10::optional<at::Tensor> & x_active_mask, const c10::optional<at::Tensor> & shared_expert_x, int64_t expert_shard_type, int64_t shared_expert_num, int64_t shared_expert_rank_num, int64_t global_bs, int64_t comm_quant_mode, int64_t comm_type, c10::string_view comm_alg)
{
    auto out_sizes = npu_moe_distribute_combine_teardown_out_size(expand_x, expert_ids);
    auto group_ep_ptr = const_cast<char *>(group_ep.data());
    auto comm_alg_ptr = const_cast<char *>(comm_alg.data());
    auto output_size_0 = out_sizes[0];
    auto output_dtype_0 = expand_x.scalar_type();
    at::Tensor out_x = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                expand_x.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMoeDistributeCombineTeardown, expand_x, quant_expand_x, expert_ids, expand_idx, expert_scales, comm_cmd_info, x_active_mask, shared_expert_x, group_ep_ptr, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type, shared_expert_num, shared_expert_rank_num, global_bs, comm_quant_mode, comm_type, comm_alg_ptr, out_x);
    return out_x;
}

::std::tuple<at::Tensor,at::Tensor> npu_moe_update_expert(const at::Tensor & expert_ids, const at::Tensor & eplb_table, const c10::optional<at::Tensor> & expert_scales, const c10::optional<at::Tensor> & pruning_threshold, const c10::optional<at::Tensor> & active_mask, int64_t local_rank_id, int64_t world_size, int64_t balance_mode)
{
    auto output_size_0 = expert_ids.sizes();
    auto output_size_1 = {expert_ids.size(0), expert_ids.size(1)};
    auto output_dtype_0 = expert_ids.scalar_type();
    auto output_dtype_1 = at::kBool;
    at::Tensor out_balanced_expert_ids = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                expert_ids.options().dtype(output_dtype_0));
    at::Tensor out_balanced_active_mask = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                expert_ids.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnMoeUpdateExpert, expert_ids, eplb_table, expert_scales, pruning_threshold, active_mask, local_rank_id, world_size, balance_mode, out_balanced_expert_ids, out_balanced_active_mask);
    return std::make_tuple(std::move(out_balanced_expert_ids), std::move(out_balanced_active_mask));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_add_rms_norm_dynamic_quant(const at::Tensor & x1, const at::Tensor & x2, const at::Tensor & gamma, const c10::optional<at::Tensor> & smooth_scale1, const c10::optional<at::Tensor> & smooth_scale2, const c10::optional<at::Tensor> & beta, double epsilon, ::std::array<bool,2> output_mask, c10::optional<at::ScalarType> y_dtype)
{
    auto output_size_0 = npu_add_rms_norm_dynamic_quant_y_size(x1, y_dtype);
    auto output_size_1 = output_mask[1] ? npu_add_rms_norm_dynamic_quant_y_size(x1, y_dtype) : at::IntArrayRef{};
    auto output_size_2 = x1.sizes();
    auto output_size_3 = reduce_lastdim_output_size(x1);
    auto output_size_4 = output_mask[1] ? reduce_lastdim_output_size(x1) : c10::SmallVector<int64_t, op_infer::SIZE>{};
    auto output_dtype_0 = npu_add_rms_norm_dynamic_quant_y_dtype(y_dtype);
    auto output_dtype_1 = x1.scalar_type();
    auto output_dtype_2 = at::kFloat;
    at::Tensor y1 = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x1.options().dtype(output_dtype_0));
    at::Tensor y2 = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                x1.options().dtype(output_dtype_0));
    at::Tensor x_out = npu_preparation::apply_tensor_without_format(output_size_2,
                                                                x1.options().dtype(output_dtype_1));
    at::Tensor scale1 = npu_preparation::apply_tensor_without_format(output_size_3,
                                                                x1.options().dtype(output_dtype_2));
    at::Tensor scale2 = npu_preparation::apply_tensor_without_format(output_size_4,
                                                                x1.options().dtype(output_dtype_2));
    EXEC_NPU_CMD(aclnnAddRmsNormDynamicQuantV2, x1, x2, gamma, smooth_scale1, smooth_scale2, beta, epsilon, output_mask, y1, y2, x_out, scale1, scale2);
    return std::make_tuple(std::move(y1), std::move(y2), std::move(x_out), std::move(scale1), std::move(scale2));
}

at::Tensor npu_quant_matmul_dequant(const at::Tensor & x, const at::Tensor & quantized_weight, const at::Tensor & weight_scale, const c10::optional<at::Tensor> & bias, const c10::optional<at::Tensor> & x_scale, const c10::optional<at::Tensor> & x_offset, const c10::optional<at::Tensor> & smooth_scale, c10::optional<c10::string_view> quant_mode)
{
    auto quant_mode_attr = quant_mode.has_value() ? const_cast<char *>(quant_mode.value().data()) : nullptr;
    auto trans = true;
    auto output_size_0 = {x.size(0), weight_scale.size(0)};
    auto output_dtype_0 = x.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnQuantMatmulDequant, x, quantized_weight, weight_scale, bias, x_scale, x_offset, smooth_scale, quant_mode_attr, trans, out);
    return out;
}

at::Tensor npu_group_quant(const at::Tensor & x, const at::Tensor & scale, const at::Tensor & group_index, const c10::optional<at::Tensor> & offset, c10::optional<at::ScalarType> dst_dtype)
{
    auto dst_type = npu_group_quant_dst_type(dst_dtype);
    auto output_size_0 = npu_group_quant_out_size(x, dst_dtype);
    auto output_dtype_0 = dst_type;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGroupQuant, x, scale, group_index, offset, dst_type, out);
    return out;
}

at::Tensor & npu_scatter_pa_cache_out(const at::Tensor & key, const at::Tensor & slot_mapping, const c10::optional<at::Tensor> & compress_lens, const c10::optional<at::Tensor> & compress_seq_offsets, const c10::optional<at::Tensor> & seq_lens, at::Tensor & key_cache)
{
    auto cache_mode = "Norm";
    auto output_size_0 = key_cache.sizes();
    auto output_dtype_0 = key_cache.scalar_type();
    npu_preparation::check_tensor({key, slot_mapping}, key_cache, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnScatterPaCache, key, key_cache, slot_mapping, compress_lens, compress_seq_offsets, seq_lens, cache_mode);
    return key_cache;
}

at::Tensor npu_swiglu(const at::Tensor & self, int64_t dim)
{
    auto output_size_0 = swiglu_backward_infershape(self, dim);
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSwiGlu, self, dim, out);
    return out;
}

at::Tensor npu_swiglu_backward(const at::Tensor & grad_output, const at::Tensor & self, int64_t dim)
{
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnSwiGluGrad, grad_output, self, dim, out);
    return out;
}

at::Tensor npu_clipped_swiglu(const at::Tensor & x, const c10::optional<at::Tensor> & group_index, int64_t dim, double alpha, double limit, double bias, bool interleaved)
{
    auto output_size_0 = swiglu_backward_infershape(x, dim);
    auto output_dtype_0 = x.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                x.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnClippedSwiglu, x, group_index, dim, alpha, limit, bias, interleaved, out);
    return out;
}

at::Tensor npu_mhc_sinkhorn_backward(const at::Tensor & grad_y, const at::Tensor & norm, const at::Tensor & sum)
{
    auto output_size_0 = grad_y.sizes();
    auto output_dtype_0 = grad_y.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_y.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMhcSinkhornBackward, grad_y, norm, sum, out);
    return out;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_cross_entropy_loss(const at::Tensor & input, const at::Tensor & target, const c10::optional<at::Tensor> & weight, c10::string_view reduction, int64_t ignore_index, double label_smoothing, double lse_square_scale_for_zloss, bool return_zloss)
{
    auto reduction_char = const_cast<char *>(reduction.data());
    auto output_size_0 = npu_cross_entropy_loss_loss_output_size(input, reduction);
    auto output_size_1 = input.sizes();
    auto output_size_2 = npu_cross_entropy_loss_zloss_output_size(input, reduction, return_zloss);
    auto output_size_3 = npu_cross_entropy_loss_lse_for_zloss_output_size(input, lse_square_scale_for_zloss);
    auto output_dtype_0 = input.scalar_type();
    at::Tensor loss = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    at::Tensor log_prob = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                input.options().dtype(output_dtype_0));
    at::Tensor zloss = npu_preparation::apply_tensor_without_format(output_size_2,
                                                                input.options().dtype(output_dtype_0));
    at::Tensor lse_for_zloss = npu_preparation::apply_tensor_without_format(output_size_3,
                                                                input.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCrossEntropyLoss, input, target, weight, reduction_char, ignore_index, label_smoothing, lse_square_scale_for_zloss, return_zloss, loss, log_prob, zloss, lse_for_zloss);
    return std::make_tuple(std::move(loss), std::move(log_prob), std::move(zloss), std::move(lse_for_zloss));
}

at::Tensor npu_cross_entropy_loss_backward(const at::Tensor & grad_loss, const at::Tensor & log_prob, const at::Tensor & target, const c10::optional<at::Tensor> & weight, const c10::optional<at::Tensor> & grad_zloss, const c10::optional<at::Tensor> & lse_for_zloss, c10::string_view reduction, int64_t ignore_index, double label_smoothing, double lse_square_scale_for_zloss)
{
    auto reduction_char = const_cast<char *>(reduction.data());
    auto output_size_0 = log_prob.sizes();
    auto output_dtype_0 = grad_loss.scalar_type();
    at::Tensor x_grad_out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_loss.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnCrossEntropyLossGrad, grad_loss, log_prob, target, weight, grad_zloss, lse_for_zloss, reduction_char, ignore_index, label_smoothing, lse_square_scale_for_zloss, x_grad_out);
    return x_grad_out;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_swish(const at::Tensor & input, int64_t num_groups, const at::Tensor & weight, const at::Tensor & bias, c10::optional<double> eps, c10::optional<double> swish_scale)
{
    auto data_format = "NCHW";
    auto eps_value = eps.value_or(1e-5);
    auto activate_swish = true;
    auto swish_scale_value = swish_scale.value_or(1.0);
    auto output_size_0 = input.sizes();
    auto output_size_1 = {input.size(0), num_groups};
    auto output_dtype_0 = input.scalar_type();
    auto output_dtype_1 = weight.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    at::Tensor mean = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                input.options().dtype(output_dtype_1));
    at::Tensor rstd = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                input.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnGroupNormSwish, input, weight, bias, num_groups, data_format, eps_value, activate_swish, swish_scale_value, out, mean, rstd);
    return std::make_tuple(std::move(out), std::move(mean), std::move(rstd));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_group_norm_swish_grad(const at::Tensor & grad, const at::Tensor & input, int64_t num_groups, const at::Tensor & weight, const at::Tensor & bias, const at::Tensor & mean, const at::Tensor & rstd, ::std::array<bool,3> grad_input_mask, c10::optional<double> swish_scale)
{
    auto data_format = "NCHW";
    auto swish_scale_value = swish_scale.value_or(1.0);
    auto dweight_require_grad = grad_input_mask[1] ? true : false;
    auto dbias_require_grad = grad_input_mask[2] ? true : false;
    auto output_size_0 = input.sizes();
    auto output_size_1 = weight.sizes();
    auto output_size_2 = bias.sizes();
    auto output_dtype_0 = input.scalar_type();
    auto output_dtype_1 = weight.scalar_type();
    auto output_dtype_2 = bias.scalar_type();
    at::Tensor grad_x = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad.options().dtype(output_dtype_0));
    at::Tensor grad_weight = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                grad.options().dtype(output_dtype_1));
    at::Tensor grad_bias = npu_preparation::apply_tensor_without_format(output_size_2,
                                                                grad.options().dtype(output_dtype_2));
    EXEC_NPU_CMD(aclnnGroupNormSwishGrad, grad, mean, rstd, input, weight, bias, num_groups, data_format, swish_scale_value, dweight_require_grad, dbias_require_grad, grad_x, grad_weight, grad_bias);
    return std::make_tuple(std::move(grad_x), std::move(grad_weight), std::move(grad_bias));
}

at::Tensor & npu_attn_softmax_(at::Tensor & self)
{
    auto dim = (int64_t)-1;
    EXEC_NPU_CMD(aclnnSoftmax, self, dim, self);
    return self;
}

at::Tensor npu_gather_sparse_index(const at::Tensor & input, const at::Tensor & index)
{
    auto dim = (int64_t)0;
    auto batch_dims = (int64_t)0;
    auto mode = (int64_t)1;
    auto output_size_0 = npu_gather_sparse_index_out_size(input, index);
    auto output_dtype_0 = input.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnGatherV3, input, dim, index, batch_dims, mode, out);
    return out;
}

at::Tensor _npu_index_add(const at::Tensor & self, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha)
{
    auto dim = (int64_t)0;
    auto mode = (int64_t)0;
    auto output_size_0 = self.sizes();
    auto output_dtype_0 = self.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                self.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnIndexAddV2, self, dim, index, source, alpha, mode, out);
    return out;
}

at::Tensor & _npu_index_add_(at::Tensor & self, const at::Tensor & index, const at::Tensor & source, const at::Scalar & alpha)
{
    auto dim = (int64_t)0;
    auto mode = (int64_t)0;
    EXEC_NPU_CMD(aclnnIndexAddV2, self, dim, index, source, alpha, mode, self);
    return self;
}

at::Tensor npu_nsa_compress(const at::Tensor & input, const at::Tensor & weight, int64_t compress_block_size, int64_t compress_stride, at::OptionalIntArrayRef actual_seq_len)
{
    auto layout = "TND";
    auto actual_seq_len_type = 0L;
    auto actual_seq_len_value = actual_seq_len.value_or(at::IntArrayRef{});
    auto output_size_0 = npu_nsa_compress_out_size(input, actual_seq_len_type, actual_seq_len, compress_block_size, compress_stride);
    auto output_dtype_0 = input.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnNsaCompress, input, weight, actual_seq_len_value, layout, compress_block_size, compress_stride, actual_seq_len_type, out);
    return out;
}

::std::tuple<at::Tensor,at::Tensor> npu_nsa_compress_grad(const at::Tensor & grad, const at::Tensor & input, const at::Tensor & weight, int64_t compress_block_size, int64_t compress_stride, at::OptionalIntArrayRef actual_seq_len)
{
    auto layout = "TND";
    auto actual_seq_len_value = actual_seq_len.value_or(at::IntArrayRef{});
    auto actual_seq_len_type = 0L;
    auto output_size_0 = input.sizes();
    auto output_size_1 = weight.sizes();
    auto output_dtype_0 = input.scalar_type();
    auto output_dtype_1 = weight.scalar_type();
    at::Tensor input_grad = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad.options().dtype(output_dtype_0));
    at::Tensor weight_grad = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                grad.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnNsaCompressGrad, grad, input, weight, actual_seq_len_value, compress_block_size, compress_stride, actual_seq_len_type, layout, input_grad, weight_grad);
    return std::make_tuple(std::move(input_grad), std::move(weight_grad));
}

at::Tensor & npu_nsa_compress_infer_out(const at::Tensor & input, const at::Tensor & weight, const at::Tensor & slot_mapping, int64_t compress_block_size, int64_t compress_stride, int64_t page_block_size, const c10::optional<at::Tensor> & block_table, at::OptionalIntArrayRef actual_seq_len, at::Tensor & cache)
{
    auto layout = "TND";
    auto actual_seq_len_value = actual_seq_len.value_or(at::IntArrayRef{});
    auto actual_seq_len_type = 1L;
    auto output_size_0 = cache.sizes();
    auto output_dtype_0 = cache.scalar_type();
    npu_preparation::check_tensor({input, weight, slot_mapping}, cache, output_dtype_0, output_size_0);
    EXEC_NPU_CMD(aclnnNsaCompressWithCache, input, weight, slot_mapping, actual_seq_len_value, block_table, layout, compress_block_size, compress_stride, actual_seq_len_type, page_block_size, cache);
    return cache;
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> npu_nsa_compress_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double scale_value, int64_t head_num, int64_t compress_block_size, int64_t compress_stride, int64_t select_block_size, int64_t select_block_count, const c10::optional<at::Tensor> & topk_mask, const c10::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_cmp_seq_kvlen, at::OptionalIntArrayRef actual_sel_seq_kvlen)
{
    auto layout = "TND";
    auto actual_seq_qlen_value = actual_seq_qlen.value_or(at::IntArrayRef{});
    auto actual_cmp_seq_kvlen_value = actual_cmp_seq_kvlen.value_or(at::IntArrayRef{});
    auto actual_sel_seq_kvlen_value = actual_sel_seq_kvlen.value_or(at::IntArrayRef{});
    auto sparse_mode = 1L;
    auto output_size_0 = {query.size(0), query.size(1), value.size(2)};
    auto output_size_1 = {query.size(0), key.size(1), select_block_count};
    auto output_size_2 = {query.size(0), query.size(1), 8L};
    auto output_dtype_0 = query.scalar_type();
    auto output_dtype_1 = at::kInt;
    auto output_dtype_2 = at::kFloat;
    at::Tensor attention_out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                query.options().dtype(output_dtype_0));
    at::Tensor topk_indices_out = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                query.options().dtype(output_dtype_1));
    at::Tensor softmax_max_out = npu_preparation::apply_tensor_without_format(output_size_2,
                                                                query.options().dtype(output_dtype_2));
    at::Tensor softmax_sum_out = npu_preparation::apply_tensor_without_format(output_size_2,
                                                                query.options().dtype(output_dtype_2));
    EXEC_NPU_CMD(aclnnNsaCompressAttention, query, key, value, atten_mask, topk_mask, actual_seq_qlen_value, actual_cmp_seq_kvlen_value, actual_sel_seq_kvlen_value, scale_value, head_num, layout, sparse_mode, compress_block_size, compress_stride, select_block_size, select_block_count, softmax_max_out, softmax_sum_out, attention_out, topk_indices_out);
    return std::make_tuple(std::move(attention_out), std::move(topk_indices_out), std::move(softmax_max_out), std::move(softmax_sum_out));
}

::std::tuple<at::Tensor,at::Tensor> npu_nsa_compress_attention_infer(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size, int64_t compress_block_size, int64_t compress_stride, c10::string_view layout, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & block_table, const c10::optional<at::Tensor> & topk_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_cmp_seq_kvlen, at::OptionalIntArrayRef actual_sel_seq_kvlen)
{
    auto layout_attr = const_cast<char *>(layout.data());
    auto actual_cmp_seq_kvlen_value = actual_cmp_seq_kvlen.value_or(at::IntArrayRef{});
    auto actual_sel_seq_kvlen_value = actual_sel_seq_kvlen.value_or(at::IntArrayRef{});
    auto sparse_mode = 0L;
    auto output_size_0 = npu_nsa_compress_attention_infer_out_size(query, value, key_value_head_num, layout);
    auto output_size_1 = npu_nsa_compress_attention_infer_topk_out_size(query, key_value_head_num, select_block_count, layout);
    auto output_dtype_0 = query.scalar_type();
    auto output_dtype_1 = at::kInt;
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                query.options().dtype(output_dtype_0));
    at::Tensor topk_indices_out = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                query.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnNsaCompressAttentionInfer, query, key, value, atten_mask, block_table, actual_seq_qlen, actual_cmp_seq_kvlen_value, actual_sel_seq_kvlen_value, topk_mask, head_num, key_value_head_num, select_block_size, select_block_count, compress_block_size, compress_stride, scale_value, layout_attr, page_block_size, sparse_mode, out, topk_indices_out);
    return std::make_tuple(std::move(out), std::move(topk_indices_out));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nsa_select_attention(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t select_block_size, int64_t select_block_count, const c10::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen)
{
    auto layout = "TND";
    auto actual_seq_qlen_value = actual_seq_qlen.value_or(at::IntArrayRef{});
    auto actual_seq_kvlen_value = actual_seq_kvlen.value_or(at::IntArrayRef{});
    auto sparse_mode = 2L;
    auto output_size_0 = {query.size(0), query.size(1), value.size(2)};
    auto output_size_1 = {query.size(0), query.size(1), 8L};
    auto output_dtype_0 = query.scalar_type();
    auto output_dtype_1 = at::kFloat;
    at::Tensor attention_out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                query.options().dtype(output_dtype_0));
    at::Tensor softmax_max_out = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                query.options().dtype(output_dtype_1));
    at::Tensor softmax_sum_out = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                query.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnNsaSelectedAttention, query, key, value, topk_indices, atten_mask, actual_seq_qlen_value, actual_seq_kvlen_value, scale_value, head_num, layout, sparse_mode, select_block_size, select_block_count, softmax_max_out, softmax_sum_out, attention_out);
    return std::make_tuple(std::move(attention_out), std::move(softmax_max_out), std::move(softmax_sum_out));
}

::std::tuple<at::Tensor,at::Tensor,at::Tensor> npu_nsa_select_attention_grad(const at::Tensor & grad, const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & attention_out, const at::Tensor & softmax_max, const at::Tensor & softmax_sum, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t select_block_size, int64_t select_block_count, const c10::optional<at::Tensor> & atten_mask, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen)
{
    auto layout = "TND";
    auto actual_seq_qlen_value = actual_seq_qlen.value_or(at::IntArrayRef{});
    auto actual_seq_kvlen_value = actual_seq_kvlen.value_or(at::IntArrayRef{});
    auto sparse_mode = 2L;
    auto output_size_0 = query.sizes();
    auto output_size_1 = key.sizes();
    auto output_size_2 = value.sizes();
    auto output_dtype_0 = query.scalar_type();
    auto output_dtype_1 = key.scalar_type();
    auto output_dtype_2 = value.scalar_type();
    at::Tensor query_grad = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad.options().dtype(output_dtype_0));
    at::Tensor key_grad = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                grad.options().dtype(output_dtype_1));
    at::Tensor value_grad = npu_preparation::apply_tensor_without_format(output_size_2,
                                                                grad.options().dtype(output_dtype_2));
    EXEC_NPU_CMD(aclnnNsaSelectedAttentionGrad, query, key, value, attention_out, grad, softmax_max, softmax_sum, topk_indices, actual_seq_qlen_value, actual_seq_kvlen_value, atten_mask, scale_value, select_block_size, select_block_count, head_num, layout, sparse_mode, query_grad, key_grad, value_grad);
    return std::make_tuple(std::move(query_grad), std::move(key_grad), std::move(value_grad));
}

at::Tensor npu_nsa_select_attention_infer(const at::Tensor & query, const at::Tensor & key, const at::Tensor & value, const at::Tensor & topk_indices, double scale_value, int64_t head_num, int64_t key_value_head_num, int64_t select_block_size, int64_t select_block_count, int64_t page_block_size, c10::string_view layout, const c10::optional<at::Tensor> & atten_mask, const c10::optional<at::Tensor> & block_table, at::OptionalIntArrayRef actual_seq_qlen, at::OptionalIntArrayRef actual_seq_kvlen)
{
    auto layout_ptr = const_cast<char *>(layout.data());
    auto sparse_mode = 0L;
    auto actual_seq_qlen_value = actual_seq_qlen.value_or(at::IntArrayRef{});
    auto actual_seq_kvlen_value = actual_seq_kvlen.value_or(at::IntArrayRef{});
    auto output_size_0 = npu_nsa_select_attention_infer_out_size(query, value, head_num, key_value_head_num, layout);
    auto output_dtype_0 = query.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                query.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnNsaSelectedAttentionInfer, query, key, value, topk_indices, atten_mask, block_table, actual_seq_qlen_value, actual_seq_kvlen_value, layout_ptr, head_num, key_value_head_num, select_block_size, select_block_count, page_block_size, scale_value, sparse_mode, out);
    return out;
}

at::Tensor npu_top_k_top_p(const at::Tensor & logits, const c10::optional<at::Tensor> & p, const c10::optional<at::Tensor> & k)
{
    auto output_size_0 = logits.sizes();
    auto output_dtype_0 = logits.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                logits.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnApplyTopKTopP, logits, p, k, out);
    return out;
}

::std::tuple<at::Tensor,at::Tensor> npu_moe_token_permute(const at::Tensor & tokens, const at::Tensor & indices, c10::optional<int64_t> num_out_tokens, bool padded_mode)
{
    auto num_out_tokens_value = num_out_tokens.value_or(0);
    auto flatten_size = indices.numel();
    auto actual_num_out_tokens = (num_out_tokens_value > 0) ? std::min(num_out_tokens_value, flatten_size) : num_out_tokens_value + flatten_size;
    auto output_size_0 = npu_moe_token_permute_out_size(tokens, indices, num_out_tokens);
    auto output_size_1 = {indices.numel()};
    auto output_dtype_0 = tokens.scalar_type();
    auto output_dtype_1 = at::kInt;
    at::Tensor permuted_tokens = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                tokens.options().dtype(output_dtype_0));
    at::Tensor sorted_indices = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                tokens.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnMoeTokenPermute, tokens, indices, actual_num_out_tokens, padded_mode, permuted_tokens, sorted_indices);
    return std::make_tuple(std::move(permuted_tokens), std::move(sorted_indices));
}

at::Tensor npu_moe_token_permute_grad_v2(const at::Tensor & grad_permuted_tokens, const at::Tensor & sorted_indices, int64_t tokens_size_0, at::ScalarType tokens_dtype, int64_t num_topK, bool padded_mode)
{
    auto output_size_0 = npu_moe_token_permute_grad_v2_out_size(tokens_size_0, grad_permuted_tokens, sorted_indices);
    auto output_dtype_0 = tokens_dtype;
    at::Tensor grad_tokens = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_permuted_tokens.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMoeTokenPermuteGrad, grad_permuted_tokens, sorted_indices, num_topK, padded_mode, grad_tokens);
    return grad_tokens;
}

at::Tensor npu_moe_token_permute_grad(const at::Tensor & tokens, const at::Tensor & grad_permuted_tokens, const at::Tensor & indices, const at::Tensor & sorted_indices, bool padded_mode)
{
    auto num_topK = (indices.dim() == 1) ? 1 : indices.size(1);
    auto output_size_0 = npu_moe_token_permute_grad_out_size(tokens, grad_permuted_tokens, indices, sorted_indices);
    auto output_dtype_0 = tokens.scalar_type();
    at::Tensor grad_tokens = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                tokens.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMoeTokenPermuteGrad, grad_permuted_tokens, sorted_indices, num_topK, padded_mode, grad_tokens);
    return grad_tokens;
}

at::Tensor npu_moe_token_unpermute(const at::Tensor & permuted_tokens, const at::Tensor & sorted_indices, const c10::optional<at::Tensor> & probs, bool padded_mode, at::OptionalIntArrayRef restore_shape)
{
    auto restore_shape_value = restore_shape.value_or(at::IntArrayRef{1});
    auto output_size_0 = npu_moe_token_unpermute_out_size(permuted_tokens, sorted_indices, probs);
    auto output_dtype_0 = permuted_tokens.scalar_type();
    at::Tensor unpermuted_tokens = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                permuted_tokens.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMoeTokenUnpermute, permuted_tokens, sorted_indices, probs, padded_mode, restore_shape_value, unpermuted_tokens);
    return unpermuted_tokens;
}

void npu_attention_to_ffn(const at::Tensor & x, const at::Tensor & session_id, const at::Tensor & micro_batch_id, const at::Tensor & layer_id, const at::Tensor & expert_ids, const at::Tensor & expert_rank_table, c10::string_view group, int64_t world_size, at::IntArrayRef ffn_token_info_table_shape, at::IntArrayRef ffn_token_data_shape, at::IntArrayRef attn_token_info_table_shape, int64_t moe_expert_num, const c10::optional<at::Tensor> & scales, const c10::optional<at::Tensor> & active_mask, int64_t quant_mode, int64_t sync_flag, int64_t ffn_start_rank_id)
{
    auto group_ptr = const_cast<char *>(group.data());
    EXEC_NPU_CMD(aclnnAttentionToFFN, x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, scales, active_mask, group_ptr, world_size, ffn_token_info_table_shape, ffn_token_data_shape, attn_token_info_table_shape, moe_expert_num, quant_mode, sync_flag, ffn_start_rank_id);
    return;
}

void npu_ffn_to_attention(const at::Tensor & x, const at::Tensor & session_ids, const at::Tensor & micro_batch_ids, const at::Tensor & token_ids, const at::Tensor & expert_offsets, const at::Tensor & actual_token_num, c10::string_view group, int64_t world_size, at::IntArrayRef token_info_table_shape, at::IntArrayRef token_data_shape, const c10::optional<at::Tensor> & attn_rank_table)
{
    auto group_ptr = const_cast<char *>(group.data());
    EXEC_NPU_CMD(aclnnFFNToAttention, x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, attn_rank_table, group_ptr, world_size, token_info_table_shape, token_data_shape);
    return;
}

::std::tuple<at::Tensor,at::Tensor> npu_moe_token_unpermute_grad(const at::Tensor & permuted_tokens, const at::Tensor & grad_unpermuted_tokens, const at::Tensor & sorted_indices, const c10::optional<at::Tensor> & probs, bool padded_mode, at::OptionalIntArrayRef restore_shape)
{
    auto restore_shape_value = restore_shape.value_or(at::IntArrayRef{1});
    auto output_size_0 = npu_moe_token_unpermute_grad_permuted_tokens_out_size(permuted_tokens, grad_unpermuted_tokens, sorted_indices, probs);
    auto output_size_1 = npu_moe_token_unpermute_grad_probs_out_size(permuted_tokens, grad_unpermuted_tokens, sorted_indices, probs);
    auto output_dtype_0 = permuted_tokens.scalar_type();
    auto output_dtype_1 = probs.has_value() && probs->defined() ? probs->scalar_type() : grad_unpermuted_tokens.scalar_type();
    at::Tensor grad_permuted_tokens = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                permuted_tokens.options().dtype(output_dtype_0));
    at::Tensor grad_probs = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                permuted_tokens.options().dtype(output_dtype_1));
    EXEC_NPU_CMD(aclnnMoeTokenUnpermuteGrad, permuted_tokens, grad_unpermuted_tokens, sorted_indices, probs, padded_mode, restore_shape_value, grad_permuted_tokens, grad_probs);
    return std::make_tuple(std::move(grad_permuted_tokens), std::move(grad_probs));
}

at::Tensor npu_masked_causal_conv1d(const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & mask)
{
    auto output_size_0 = input.sizes();
    auto output_dtype_0 = input.scalar_type();
    at::Tensor out = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                input.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMaskedCausalConv1d, input, weight, mask, out);
    return out;
}

::std::tuple<at::Tensor,at::Tensor> npu_masked_causal_conv1d_backward(const at::Tensor & grad_output, const at::Tensor & input, const at::Tensor & weight, const c10::optional<at::Tensor> & mask)
{
    auto output_size_0 = grad_output.sizes();
    auto output_size_1 = weight.sizes();
    auto output_dtype_0 = grad_output.scalar_type();
    at::Tensor grad_input = npu_preparation::apply_tensor_without_format(output_size_0,
                                                                grad_output.options().dtype(output_dtype_0));
    at::Tensor grad_weight = npu_preparation::apply_tensor_without_format(output_size_1,
                                                                grad_output.options().dtype(output_dtype_0));
    EXEC_NPU_CMD(aclnnMaskedCausalConv1dBackward, grad_output, input, weight, mask, grad_input, grad_weight);
    return std::make_tuple(std::move(grad_input), std::move(grad_weight));
}

}  // namespace op_api
