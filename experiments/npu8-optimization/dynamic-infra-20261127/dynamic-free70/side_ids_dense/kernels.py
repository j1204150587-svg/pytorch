"""Prepare dynamic side-table IDs with continuous reads and writes before the native transpose."""
import triton
import triton.language as tl


@triton.jit
def dense_side_ids(ids, offsets, packed):
    token = tl.program_id(0)
    half = tl.program_id(1)
    lane = tl.arange(0, 2048)
    position = token * 4096 + half * 2048 + lane
    value = tl.load(ids + position).to(tl.int64)
    local_table = half * 512 + lane // 4
    base = tl.load(offsets + local_table)
    valid = (value >= 0) & (value < 64)
    clean = tl.where(valid, value, 0).to(tl.uint64)
    result = tl.where(valid, (clean + base.to(tl.uint64)).to(tl.int64, bitcast=True), 65536)
    tl.store(packed + position, result)
