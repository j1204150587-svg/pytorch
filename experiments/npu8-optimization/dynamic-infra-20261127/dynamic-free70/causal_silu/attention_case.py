import torch
import torch.nn.functional as F


def original(q, k, v):
    l = q.size(-2)
    scores = torch.matmul(q, k.transpose(-2, -1)) * .25
    causal = torch.tril(torch.ones(l, l, dtype=torch.bool, device=q.device))
    scores = F.silu(scores).masked_fill(~causal, 0.)
    scores = scores / float(l)
    return torch.matmul(scores, v)


def layout(q, k, v):
    if q.ndim != 4:
        return False
    b, h, l, d = q.shape
    return (1 <= b <= 4 and h == 4 and 12 <= l <= 32 and d == 16
            and all(x.shape == q.shape and x.dtype == torch.bfloat16 and x.device == q.device
                    and x.stride(1) == 16 and x.stride(2) == 256 and x.stride(3) == 1
                    and (b == 1 or x.stride(0) == l*256) for x in (q, k, v)))


def cpu_full_forward(q, k, v):
    raw = (q.float() @ k.float().transpose(-1, -2)).bfloat16()
    x = (raw.float() * .25).bfloat16()
    sig = torch.sigmoid(x.float())
    a = (x.float() * sig).bfloat16()
    s = (torch.tril(a).float() / float(q.size(2))).bfloat16()
    y = (s.float() @ v.float()).bfloat16()
    return x, s, y


class Full(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v):
        b, h, l, d = q.shape
        if q.device.type == "cpu":
            x, s, y = cpu_full_forward(q, k, v)
        else:
            from attn_kernels import attention_forward
            x = torch.empty((b, h, l, l), device=q.device, dtype=q.dtype)
            s = torch.empty_like(x)
            y = torch.empty(q.shape, device=q.device, dtype=q.dtype)
            attention_forward[(b*h,)](q, k, v, x, s, y, l, enable_fp_fusion=False)
        ctx.save_for_backward(q, k, v, x, s)
        return y

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        q, k, v, x, s = ctx.saved_tensors
        b, h, l, d = q.shape
        if q.device.type == "cpu":
            dv = (s.float().transpose(-2, -1) @ dy.float()).bfloat16()
            ds = (dy.float() @ v.float().transpose(-2, -1)).bfloat16()
            g = torch.tril((ds.float() / float(l)).bfloat16()).float()
            sig = torch.sigmoid(x.float())
            dx = (g * (sig * (1. + x.float() * (1. - sig)))).bfloat16()
            draw = (dx.float() * .25).bfloat16().float()
            return ((draw @ k.float()).bfloat16(), (draw.transpose(-1, -2) @ q.float()).bfloat16(), dv)
        from attn_kernels import attention_backward
        dq, dk, dv = [torch.empty(q.shape, device=q.device, dtype=q.dtype) for _ in range(3)]
        attention_backward[(b*h,)](q, k, v, x, s, dy, dq, dk, dv, l, *dy.stride(), enable_fp_fusion=False)
        return dq, dk, dv


def full(q, k, v):
    return Full.apply(q, k, v) if layout(q, k, v) else original(q, k, v)


def mask_div_impl(x):
    l = x.size(-1)
    if x.device.type == "cpu":
        keep = torch.ones(l, l, dtype=torch.bool, device=x.device).tril()
        return torch.where(keep, x.float() / float(l), 0.).bfloat16()
    from attn_kernels import mask_div
    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    n = x.numel()
    mask_div[((n+255)//256,)](x, out, l, n, *x.stride(), 256)
    return out


class MaskDiv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return mask_div_impl(x)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, dy):
        return mask_div_impl(dy)


def partial(q, k, v):
    if not layout(q, k, v):
        return original(q, k, v)
    scores = torch.matmul(q, k.transpose(-2, -1)) * .25
    scores = MaskDiv.apply(F.silu(scores))
    return torch.matmul(scores, v)
