"""Row norms and nonfinite flags for a contiguous BF16 gradient partition."""
import torch
import triton
import triton.language as tl


@triton.jit
def row_moments(data, offsets, lengths, norms, bad, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    start = tl.load(offsets + row)
    n = tl.load(lengths + row)
    lane = tl.arange(0, BLOCK)
    squares = tl.full((BLOCK,), 0, tl.float32)
    invalid = tl.full((BLOCK,), 0, tl.int32)
    for part in range(0, tl.cdiv(n, BLOCK)):
        index = part * BLOCK + lane
        raw = tl.load(data + start + index, mask=index < n, other=0.0)
        bits = raw.to(tl.uint16, bitcast=True)
        val = raw.to(tl.float32)
        squares += val * val
        invalid |= ((bits & 0x7f80) == 0x7f80).to(tl.int32)
    tl.store(norms + row, tl.sqrt(tl.sum(squares, 0)))
    tl.store(bad + row, tl.max(invalid, 0))


class PartitionMoments:
    def __init__(self, sizes, device):
        starts = []
        offset = 0
        for n in sizes:
            if n <= 0:
                raise ValueError("expected nonempty gradients")
            starts.append(offset)
            offset += n
        if not sizes or offset >= 2**31:
            raise ValueError("partition exceeds the supported offset range")
        self.count = len(sizes)
        self.numel = offset
        self.offsets = torch.tensor(starts, dtype=torch.int32).to(device)
        self.sizes = torch.tensor(sizes, dtype=torch.int32).to(device)
        self.cached = None

    def compute(self, flat):
        if flat.dtype != torch.bfloat16 or flat.shape != (self.numel,) or not flat.is_contiguous():
            raise ValueError("expected flat contiguous BF16")
        norm = torch.empty(self.count, dtype=torch.float32, device=flat.device)
        bad = torch.empty(self.count, dtype=torch.int32, device=flat.device)
        row_moments[(self.count,)](flat, self.offsets, self.sizes, norm, bad, BLOCK=1024)
        return norm, bad

    def scan(self, flat):
        norms, bad = self.compute(flat)
        self.cached = (flat, flat._version, torch.npu.current_stream(), norms)
        return bad.amax(dim=0, keepdim=True).ne(0)

    def square(self, flat):
        snap = self.cached
        self.cached = None
        if snap is None or snap[0] is not flat or snap[1] != flat._version or snap[2] != torch.npu.current_stream():
            return None
        return snap[3].square().sum().float()
