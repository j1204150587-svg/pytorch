import triton
import triton.language as tl


@triton.jit
def causal_score_forward(raw, table, out):
    row = tl.program_id(0)
    batch = tl.program_id(1)
    length = tl.num_programs(0)
    lane = tl.arange(0, 128)
    col = lane % 32
    pos = ((batch * 4 + lane // 32) * length + row) * length + col
    x = (tl.load(raw + pos, mask=col < length, other=0).to(tl.float32) * 0.25).to(tl.bfloat16)
    code = x.to(tl.int16, bitcast=True).to(tl.int32) & 65535
    y = tl.load(table + code).to(tl.float32)
    value = tl.where(row >= col, y, 0.0) / length.to(tl.float32)
    tl.store(out + pos, value.to(tl.bfloat16), mask=col < length)


@triton.jit
def causal_score_backward(raw, grad, slopes, out):
    row = tl.program_id(0)
    batch = tl.program_id(1)
    length = tl.num_programs(0)
    lane = tl.arange(0, 128)
    col = lane % 32
    pos = ((batch * 4 + lane // 32) * length + row) * length + col
    x = (tl.load(raw + pos, mask=col < length, other=0).to(tl.float32) * 0.25).to(tl.bfloat16)
    code = x.to(tl.int16, bitcast=True).to(tl.int32) & 65535
    slope = tl.load(slopes + code)
    dy = tl.load(grad + pos, mask=col < length, other=0).to(tl.float32)
    masked = tl.where(row >= col, (dy / length.to(tl.float32)).to(tl.bfloat16).to(tl.float32), 0.0)
    dx = (masked * slope).to(tl.bfloat16).to(tl.float32) * 0.25
    tl.store(out + pos, dx.to(tl.bfloat16), mask=col < length)
