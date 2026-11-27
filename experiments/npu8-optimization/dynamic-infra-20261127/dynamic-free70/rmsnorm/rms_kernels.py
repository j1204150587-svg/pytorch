"""Elementwise pieces of the eager BF16 RMSNorm; original ATen reductions remain in use."""
import triton
import triton.language as tl


@triton.jit
def scale_gain(x, inv, weight, intermediate, out):
    row = tl.program_id(0)
    feature = tl.arange(0, 64)
    offset = row * 64 + feature
    a = tl.load(x+offset).to(tl.float32)
    r = tl.load(inv+row).to(tl.float32)
    w = tl.load(weight+feature).to(tl.float32)
    z = (a * r).to(tl.bfloat16)
    tl.store(intermediate+offset,z)
    tl.store(out+offset,(z.to(tl.float32) * w).to(tl.bfloat16))


@triton.jit
def scale_gain_backward(x, inv, weight, intermediate, dy, dx, inv_terms, weight_terms):
    row = tl.program_id(0)
    feature = tl.arange(0,64)
    offset = row * 64 + feature
    g = tl.load(dy+offset).to(tl.float32)
    w = tl.load(weight+feature).to(tl.float32)
    a = tl.load(x+offset).to(tl.float32)
    r = tl.load(inv+row).to(tl.float32)
    z = tl.load(intermediate+offset).to(tl.float32)
    gz = (g*w).to(tl.bfloat16).to(tl.float32)
    tl.store(dx+offset,(gz*r).to(tl.bfloat16))
    tl.store(inv_terms+offset,(gz*a).to(tl.bfloat16))
    tl.store(weight_terms+offset,(g*z).to(tl.bfloat16))
