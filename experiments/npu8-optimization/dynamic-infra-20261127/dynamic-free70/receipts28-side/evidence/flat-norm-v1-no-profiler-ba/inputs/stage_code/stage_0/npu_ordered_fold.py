"""Ordered BF16 accumulation for actual [T,B,L,64] values on Ascend."""
import triton
import triton.language as tl


@triton.jit
def ordered_64(source, result, tables: tl.constexpr):
    # Each program owns one complete [D=64] token. The logical grid has B*L
    # programs for this call. Both the grid and source use the actual shape.
    column = tl.program_id(0) * 64 + tl.arange(0, 64)
    width = tl.num_programs(0) * 64
    total = tl.load(source + column).to(tl.float32)
    for table in range(1, tables):
        value = tl.load(source + table * width + column).to(tl.float32)
        total = (total + value).to(tl.bfloat16).to(tl.float32)
    tl.store(result + column, total.to(tl.bfloat16))
