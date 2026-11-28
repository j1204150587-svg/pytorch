"""Batch equal-shape parameter norms and restore the optimizer's reduction order."""
from collections import defaultdict

import torch


def norm_without_casts(full, groups):
    values = []
    for offset, shape, count, width in groups:
        part = full.narrow(0, offset, count * width)
        if count == 1:
            value = torch.linalg.vector_norm(part.view(shape), ord=2).reshape(1)
        else:
            value = torch.linalg.vector_norm(part.view(count, width), ord=2, dim=1)
        values.append(value)
    norms = values[0] if len(values) == 1 else torch.cat(values)
    return norms.square().sum().float()


class ParamNormGroups:
    def __init__(self, plan, device):
        groups = defaultdict(list)
        total = count = 0
        for offset, shape, rows, width in plan:
            if offset != total or rows <= 0 or width <= 0:
                raise ValueError("expected the full contiguous gradient partition")
            for index in range(rows):
                groups[tuple(shape)].append((count, offset + index * width))
                count += 1
            total += rows * width
        self.total = total
        self.layout = []
        self.extra_indices = []
        self.return_order = [0] * count
        taken = gathered = 0
        for shape, members in groups.items():
            width = torch.Size(shape).numel()
            first = members[0][1]
            rows = len(members)
            contiguous = all(offset == first + i * width for i, (_, offset) in enumerate(members))
            start = first if contiguous else gathered
            if not contiguous:
                for _, offset in members:
                    self.extra_indices.append(torch.arange(offset, offset + width, dtype=torch.int64))
                gathered += rows * width
            for i, (orig, _) in enumerate(members):
                self.return_order[orig] = taken + i
            self.layout.append((shape, width, rows, start, contiguous))
            taken += rows
        self.gather = torch.cat(self.extra_indices).to(device) if self.extra_indices else None
        self.order = torch.tensor(self.return_order, dtype=torch.int64, device=device)
        self.extra_indices.clear()
        self.gather_elements = gathered

    def __call__(self, full):
        # The pinned 910_9362 wheel maps .double() to a real FP32 tensor.
        supported = (torch.float32, torch.float64) if full.device.type == "npu" else (torch.float64,)
        if full.ndim != 1 or full.dtype not in supported or full.numel() != self.total:
            raise ValueError("expected the same wide gradient partition")
        packed = full.index_select(0, self.gather) if self.gather is not None else full
        pieces = []
        for shape, width, rows, start, contiguous in self.layout:
            source = full if contiguous else packed
            part = source.narrow(0, start, width * rows)
            if rows == 1:
                norm = torch.linalg.vector_norm(part.view(shape), ord=2).reshape(1)
            else:
                norm = torch.linalg.vector_norm(part.view(rows, width), ord=2, dim=1)
            pieces.append(norm)
        ordered = torch.cat(pieces).index_select(0, self.order)
        return ordered.square().sum().float()
