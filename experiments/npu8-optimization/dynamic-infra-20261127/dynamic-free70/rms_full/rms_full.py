"""Eager whole-row BF16 RMSNorm, with native weight-gradient reduction."""
import torch


FORWARD = ("sq", "mean", "var_eps", "inv", "z", "y")
BACKWARD = ("gz", "direct", "inv_terms", "grad_inv", "inv3", "half_grad",
            "grad_var", "grad_sq", "double_x", "from_var", "dx", "dw_terms")


def stock(x, w, epsilon=1e-6):
    return (x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)) * w


def aten_forward(x, w, epsilon):
    sq = x.pow(2)
    mean = sq.mean(dim=-1, keepdim=True)
    var_eps = mean + epsilon
    inv = torch.rsqrt(var_eps)
    z = x * inv
    return dict(zip(FORWARD, (sq, mean, var_eps, inv, z, z * w)))


def aten_backward(x, w, inv, z, dy):
    gz = dy * w
    direct = gz * inv
    inv_terms = gz * x
    grad_inv = inv_terms.sum_to_size(inv.shape)
    inv3 = inv.pow(3)
    half_grad = -0.5 * grad_inv
    grad_var = half_grad * inv3
    grad_sq = grad_var.expand_as(x) / 64
    double_x = 2 * x
    from_var = grad_sq * double_x
    values = (gz, direct, inv_terms, grad_inv, inv3, half_grad, grad_var, grad_sq,
              double_x, from_var, direct + from_var, dy * z)
    return dict(zip(BACKWARD, values))


def _trace(x, stages, debug):
    # A valid pointer placeholder allows the same JIT signature on the fast leg.
    return x.new_empty((*x.shape[:-1], stages, 64)) if debug else x


def forward_row(x, w, epsilon, debug=False):
    if x.device.type == "cpu":
        values = aten_forward(x, w, epsilon)
        return values["y"], values["inv"], values["z"], values if debug else {}
    from rms_full_kernels import full_forward
    inv = x.new_empty((*x.shape[:-1], 1))
    z, y = torch.empty_like(x), torch.empty_like(x)
    trace = _trace(x, len(FORWARD), debug)
    full_forward[(x.numel() // 64,)](x, w, inv, z, y, trace, epsilon, debug)
    stages = dict(zip(FORWARD, trace.unbind(-2))) if debug else {}
    return y, inv, z, stages


def backward_row(x, w, inv, z, dy, debug=False):
    if x.device.type == "cpu":
        values = aten_backward(x, w, inv, z, dy)
        return values["dx"], values["dw_terms"], values if debug else {}
    from rms_full_kernels import full_backward
    dx, dw_terms = torch.empty_like(x), torch.empty_like(x)
    trace = _trace(x, len(BACKWARD), debug)
    full_backward[(x.numel() // 64,)](x, w, inv, z, dy.contiguous(), dx, dw_terms, trace, debug)
    stages = dict(zip(BACKWARD, trace.unbind(-2))) if debug else {}
    return dx, dw_terms, stages


class FullRMS(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, epsilon):
        y, inv, z, _ = forward_row(x, w, epsilon)
        ctx.save_for_backward(x, w, inv, z)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad):
        x, w, inv, z = ctx.saved_tensors
        dx, dw_terms, _ = backward_row(x, w, inv, z, grad)
        return dx, dw_terms.sum_to_size(w.shape), None


def supports(x, w):
    return (x.ndim == 3 and x.shape[-1] == 64 and w.shape == (64,)
            and x.dtype == w.dtype == torch.bfloat16 and x.device == w.device
            and x.is_contiguous() and w.is_contiguous() and torch.is_grad_enabled())


def candidate(x, w, epsilon=1e-6):
    return FullRMS.apply(x, w, epsilon) if supports(x, w) else stock(x, w, epsilon)
