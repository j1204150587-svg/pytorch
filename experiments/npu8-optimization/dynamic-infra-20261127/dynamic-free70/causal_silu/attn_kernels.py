"""HSTU attention for real [B,4,L,16] projection views, 12 <= L <= 32."""
import triton
import triton.language as tl


@triton.jit
def attention_forward(Q, K, V, X, S, Y, L):
    bh = tl.program_id(0)
    at = tl.arange(0, 32)
    dims = tl.arange(0, 16)
    valid_row = at[:, None] < L
    # Each tensor points at its own slice of the common [B,L,256] projection.
    qkv_offsets = (bh // 4 * L * 256 + bh % 4 * 16 +
                   at[:, None] * 256 + dims[None, :])
    q = tl.load(Q + qkv_offsets, valid_row, 0)
    k = tl.load(K + qkv_offsets, valid_row, 0)
    v = tl.load(V + qkv_offsets, valid_row, 0)
    raw = tl.dot(q, tl.trans(k)).to(tl.bfloat16)
    x = (raw.to(tl.float32) * 0.25).to(tl.bfloat16)
    f = x.to(tl.float32)
    activated = (f * tl.sigmoid(f)).to(tl.bfloat16)
    causal = (at[:, None] >= at[None, :]) & valid_row & (at[None, :] < L)
    zeroed = tl.where(causal, activated.to(tl.float32), 0.0)
    score = (zeroed / L.to(tl.float32)).to(tl.bfloat16)
    square = bh * L * L + at[:, None] * L + at[None, :]
    valid = valid_row & (at[None, :] < L)
    tl.store(X + square, x, valid)
    tl.store(S + square, score, valid)
    y = tl.dot(score, v).to(tl.bfloat16)
    tl.store(Y + (bh * L + at[:, None]) * 16 + dims[None, :], y, valid_row)


@triton.jit
def attention_backward(Q, K, V, X, S, DY, DQ, DK, DV, L, G0, G1, G2, G3):
    bh = tl.program_id(0)
    at = tl.arange(0, 32)
    dims = tl.arange(0, 16)
    valid_row = at[:, None] < L
    qkv_offsets = (bh // 4 * L * 256 + bh % 4 * 16 +
                   at[:, None] * 256 + dims[None, :])
    q = tl.load(Q + qkv_offsets, valid_row, 0)
    k = tl.load(K + qkv_offsets, valid_row, 0)
    v = tl.load(V + qkv_offsets, valid_row, 0)
    gout = bh // 4 * G0 + bh % 4 * G1 + at[:, None] * G2 + dims[None, :] * G3
    dy = tl.load(DY + gout, valid_row, 0)
    square = bh * L * L + at[:, None] * L + at[None, :]
    valid = valid_row & (at[None, :] < L)
    x = tl.load(X + square, valid, 0).to(tl.float32)
    s = tl.load(S + square, valid, 0)
    dv = tl.dot(tl.trans(s), dy).to(tl.bfloat16)
    ds = tl.dot(dy, tl.trans(v)).to(tl.bfloat16)
    divided = (ds.to(tl.float32) / L.to(tl.float32)).to(tl.bfloat16)
    grad = tl.where((at[:, None] >= at[None, :]) & valid, divided.to(tl.float32), 0.0)
    sigmoid = tl.sigmoid(x)
    dx = (grad * (sigmoid * (1.0 + x * (1.0 - sigmoid)))).to(tl.bfloat16)
    draw = (dx.to(tl.float32) * 0.25).to(tl.bfloat16)
    dq = tl.dot(draw, k).to(tl.bfloat16)
    dk = tl.dot(tl.trans(draw), q).to(tl.bfloat16)
    out = (bh * L + at[:, None]) * 16 + dims[None, :]
    tl.store(DQ + out, dq, valid_row)
    tl.store(DK + out, dk, valid_row)
    tl.store(DV + out, dv, valid_row)


@triton.jit
def mask_div(X, Y, L, N, S0, S1, S2, S3, WIDTH: tl.constexpr):
    i = tl.program_id(0) * WIDTH + tl.arange(0, WIDTH)
    c = i % L
    r = i // L % L
    h = i // (L * L) % 4
    b = i // (4 * L * L)
    x = tl.load(X + b*S0 + h*S1 + r*S2 + c*S3, i < N, 0)
    y = tl.where(c <= r, x.to(tl.float32) / L.to(tl.float32), 0.0).to(tl.bfloat16)
    tl.store(Y + i, y, i < N)
