"""Match each intermediate BF16 rounding of the HSTU RMSNorm."""
import torch


def stock(x, weight, epsilon=1e-6):
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)) * weight


def stock_npu(x, weight, epsilon=1e-6):
    import torch_npu
    return torch_npu.npu_rms_norm(x, weight, epsilon=epsilon)[0]


class ScaleGain(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, inv, weight):
        if x.device.type == "cpu":
            z = x * inv
            result = z * weight
        else:
            from rms_kernels import scale_gain
            z, result = torch.empty_like(x), torch.empty_like(x)
            scale_gain[(x.numel()//64,)](x,inv,weight,z,result)
        ctx.save_for_backward(x,inv,weight,z)
        return result

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        x,inv,weight,z = ctx.saved_tensors
        if x.device.type == "cpu":
            gz = grad * weight
            dx, inv_terms, weight_terms = gz*inv, gz*x, grad*z
        else:
            from rms_kernels import scale_gain_backward
            dx, inv_terms, weight_terms = (torch.empty_like(x) for _ in range(3))
            scale_gain_backward[(x.numel()//64,)](x,inv,weight,z,grad.contiguous(),dx,inv_terms,weight_terms)
        return dx, inv_terms.sum_to_size(inv.shape), weight_terms.sum_to_size(weight.shape)


def partial(x, weight, epsilon=1e-6):
    if (x.ndim != 3 or x.shape[-1] != 64 or weight.shape != (64,)
            or x.dtype != torch.bfloat16 or weight.dtype != x.dtype or x.device != weight.device
            or not x.is_contiguous() or not weight.is_contiguous() or not torch.is_grad_enabled()):
        return stock(x,weight,epsilon)
    inv = torch.rsqrt(x.pow(2).mean(dim=-1,keepdim=True) + epsilon)
    return ScaleGain.apply(x,inv,weight)
