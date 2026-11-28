import triton
import triton.language as tl


@triton.jit
def rms_grad_products(x, w, inv, z, dy, direct, row_products, weight_products):
    row = tl.program_id(0)
    col = tl.arange(0, 64)
    pos = row * 64 + col
    g = tl.load(dy + pos).to(tl.float32)
    weight = tl.load(w + col).to(tl.float32)
    gz = (g * weight).to(tl.bfloat16).to(tl.float32)
    tl.store(direct + pos, gz * tl.load(inv + row).to(tl.float32))
    tl.store(row_products + pos, gz * tl.load(x + pos).to(tl.float32))
    tl.store(weight_products + pos, g * tl.load(z + pos).to(tl.float32))


@triton.jit
def rms_grad_variance(x, dr, inv_cubed, result):
    row = tl.program_id(0)
    pos = row * 64 + tl.arange(0, 64)
    negative_half = (tl.load(dr + row).to(tl.float32) * -0.5).to(tl.bfloat16)
    dvar = (negative_half.to(tl.float32) * tl.load(inv_cubed + row).to(tl.float32)).to(tl.bfloat16)
    dsq = (dvar.to(tl.float32) * 0.015625).to(tl.bfloat16)
    dx = (tl.load(x + pos).to(tl.float32) * 2.0).to(tl.bfloat16)
    tl.store(result + pos, dsq.to(tl.float32) * dx.to(tl.float32))
