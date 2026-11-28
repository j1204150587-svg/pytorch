import triton
import triton.language as tl


@triton.jit
def causal_score_forward(raw, table, out):
    length = tl.num_programs(0)
    batch = tl.num_programs(1)
    tile = tl.program_id(1) * length + tl.program_id(0)
    pos = tile * 128 + tl.arange(0,128)
    active = pos < batch * 4 * length * length
    row, col = pos // length % length, pos % length
    x = (tl.load(raw + pos, mask=active, other=0).to(tl.float32) * 0.25).to(tl.bfloat16)
    code = x.to(tl.int16, bitcast=True).to(tl.int32) & 65535
    y = tl.load(table + code).to(tl.float32)
    value = tl.where(row >= col, y, 0.0) / length.to(tl.float32)
    tl.store(out + pos, value.to(tl.bfloat16), mask=active)


@triton.jit
def causal_score_backward(raw, grad, slopes, out):
    length = tl.num_programs(0)
    batch = tl.num_programs(1)
    tile = tl.program_id(1) * length + tl.program_id(0)
    pos = tile * 128 + tl.arange(0,128)
    active = pos < batch * 4 * length * length
    row, col = pos // length % length, pos % length
    x = (tl.load(raw + pos, mask=active, other=0).to(tl.float32) * 0.25).to(tl.bfloat16)
    code = x.to(tl.int16, bitcast=True).to(tl.int32) & 65535
    slope = tl.load(slopes + code)
    dy = tl.load(grad + pos, mask=active, other=0).to(tl.float32)
    masked = tl.where(row >= col, (dy / length.to(tl.float32)).to(tl.bfloat16).to(tl.float32), 0.0)
    dx = (masked * slope).to(tl.bfloat16).to(tl.float32) * 0.25
    tl.store(out + pos, dx.to(tl.bfloat16), mask=active)
