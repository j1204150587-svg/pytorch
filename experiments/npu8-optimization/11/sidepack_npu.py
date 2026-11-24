"""Side-table batching with an ordered BF16 fold on NPU."""

import torch
import triton
import triton.language as tl


@triton.jit
def fold(source, result, width: tl.constexpr, tables: tl.constexpr, block: tl.constexpr):
    column = tl.program_id(0) * block + tl.arange(0, block)
    accumulator = tl.load(source + column).to(tl.float32)
    for row in range(1, tables):
        value = tl.load(source + row * width + column).to(tl.float32)
        accumulator = (accumulator + value).to(tl.bfloat16).to(tl.float32)
    tl.store(result + column, accumulator.to(tl.bfloat16))


class OrderedFold(torch.autograd.Function):
    @staticmethod
    def forward(context, values):
        context.shape = values.shape
        values = values.contiguous()
        result = torch.empty(values.shape[1:], dtype=values.dtype, device=values.device)
        width = result.numel()
        block = 128
        if values.dtype != torch.bfloat16 or width % block:
            raise RuntimeError("ordered NPU fold requires BF16 and aligned result size")
        fold[(width // block,)](values, result, width, values.shape[0], block)
        return result

    @staticmethod
    def backward(context, grad):
        return grad.unsqueeze(0).expand(context.shape)


def side_features(tables, feature_ids, offsets, table_rows):
    weight = torch.cat([table.weight for table in tables], dim=0)
    table_ids = feature_ids.movedim(2, 0)
    valid = (table_ids >= 0) & (table_ids < table_rows)
    rows = torch.where(valid, table_ids + offsets, -1).contiguous()
    values = torch.nn.functional.embedding(rows, weight).mean(dim=-2)
    return OrderedFold.apply(values)
