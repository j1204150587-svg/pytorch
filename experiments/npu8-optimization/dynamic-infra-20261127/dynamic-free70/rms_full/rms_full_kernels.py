"""Whole-row BF16 RMSNorm. A program owns precisely one real 64-wide row."""
import triton
import triton.language as tl


@triton.jit
def full_forward(x, w, inv_out, z_out, y_out, trace, EPS: tl.constexpr, DEBUG: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 64)
    off = row * 64 + col
    a = tl.load(x + off).to(tl.float32)
    weight = tl.load(w + col).to(tl.float32)
    sq = (a * a).to(tl.bfloat16).to(tl.float32)
    mean = (tl.sum(sq, axis=0) * (1.0 / 64)).to(tl.bfloat16).to(tl.float32)
    var_eps = (mean + EPS).to(tl.bfloat16).to(tl.float32)
    inv = tl.rsqrt(var_eps).to(tl.bfloat16).to(tl.float32)
    z = (a * inv).to(tl.bfloat16).to(tl.float32)
    y = (z * weight).to(tl.bfloat16)
    tl.store(inv_out + row, inv)
    tl.store(z_out + off, z)
    tl.store(y_out + off, y)
    if DEBUG:
        base = trace + row * 6 * 64 + col
        tl.store(base, sq)
        tl.store(base + 64, mean)
        tl.store(base + 128, var_eps)
        tl.store(base + 192, inv)
        tl.store(base + 256, z)
        tl.store(base + 320, y)


@triton.jit
def full_backward(x, w, inv_in, z_in, dy_in, dx_out, dw_terms_out, trace, DEBUG: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, 64)
    off = row * 64 + col
    a = tl.load(x + off).to(tl.float32)
    weight = tl.load(w + col).to(tl.float32)
    inv = tl.load(inv_in + row).to(tl.float32)
    z = tl.load(z_in + off).to(tl.float32)
    dy = tl.load(dy_in + off).to(tl.float32)
    gz = (dy * weight).to(tl.bfloat16).to(tl.float32)
    direct = (gz * inv).to(tl.bfloat16).to(tl.float32)
    inv_terms = (gz * a).to(tl.bfloat16).to(tl.float32)
    grad_inv = tl.sum(inv_terms, axis=0).to(tl.bfloat16).to(tl.float32)
    inv3 = (inv * inv * inv).to(tl.bfloat16).to(tl.float32)
    half_grad = (-0.5 * grad_inv).to(tl.bfloat16).to(tl.float32)
    grad_var = (half_grad * inv3).to(tl.bfloat16).to(tl.float32)
    grad_sq = (grad_var * (1.0 / 64)).to(tl.bfloat16).to(tl.float32)
    double_x = (2.0 * a).to(tl.bfloat16).to(tl.float32)
    from_var = (grad_sq * double_x).to(tl.bfloat16).to(tl.float32)
    dx = (direct + from_var).to(tl.bfloat16)
    dw_terms = (dy * z).to(tl.bfloat16)
    tl.store(dx_out + off, dx)
    tl.store(dw_terms_out + off, dw_terms)
    if DEBUG:
        base = trace + row * 12 * 64 + col
        tl.store(base, gz)
        tl.store(base + 64, direct)
        tl.store(base + 128, inv_terms)
        tl.store(base + 192, grad_inv)
        tl.store(base + 256, inv3)
        tl.store(base + 320, half_grad)
        tl.store(base + 384, grad_var)
        tl.store(base + 448, grad_sq)
        tl.store(base + 512, double_x)
        tl.store(base + 576, from_var)
        tl.store(base + 640, dx)
        tl.store(base + 704, dw_terms)
