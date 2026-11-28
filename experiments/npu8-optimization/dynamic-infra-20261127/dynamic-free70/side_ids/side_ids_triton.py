"""One dynamic HSTU side-id packing kernel; [B,L,1024,4] -> [1024,B,L,4]."""
import triton
import triton.language as tl


@triton.jit
def side_ids_kernel(ids, offsets, packed):
    token = tl.program_id(0)
    tokens = tl.num_programs(0)
    group = tl.program_id(1)
    lane = tl.arange(0, 256)
    table = group * 64 + lane // 4
    feature = lane % 4
    value = tl.load(ids + token * 4096 + table * 4 + feature).to(tl.int64)
    base = tl.load(offsets + table)
    valid = (value >= 0) & (value < 64)
    # Match torch int64 modular addition even when a caller supplies extreme offsets.
    summed = tl.where(valid, value, 0).to(tl.uint64) + base.to(tl.uint64)
    result = tl.where(valid, summed.to(tl.int64, bitcast=True), 65536)
    tl.store(packed + (table * tokens + token) * 4 + feature, result)
