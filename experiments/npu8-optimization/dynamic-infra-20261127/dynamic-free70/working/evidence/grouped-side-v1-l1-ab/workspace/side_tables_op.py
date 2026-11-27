"""Experimental eager dispatcher primitive for dense independent NPU side embeddings."""
import torch
import torch.nn.functional as F


class OrderedTableSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, means):
        if means.ndim != 4 or means.dtype != torch.bfloat16 or means.shape[-1] != 64:
            raise ValueError("expected [T,B,L,64] BF16 means")
        ctx.shape = means.shape
        if means.device.type == "cpu":
            # The CPU oracle exercises logical table identity and autograd.
            total = means[0]
            for value in means[1:].unbind(0):
                total = total + value
            return total
        if means.device.type != "npu":
            raise ValueError("this primitive supports NPU and the CPU reference")
        from npu_ordered_fold import ordered_64
        result = torch.empty(means.shape[1:], device=means.device, dtype=means.dtype)
        ordered_64[(means.shape[1] * means.shape[2],)](means.contiguous(), result, means.shape[0])
        return result

    @staticmethod
    def backward(ctx, grad):
        # This is the Jacobian of the reference chain of ATen BF16 additions.
        return grad.unsqueeze(0).expand(ctx.shape)


def grouped_side_sum(weights, ids, offsets):
    rows, width = weights[0].shape
    tables = len(weights)
    if (ids.ndim != 4 or ids.shape[2] != tables or ids.dtype not in (torch.int32, torch.int64)
            or offsets.shape != (tables, 1, 1, 1) or offsets.dtype != torch.int64
            or width != 64 or weights[0].dtype != torch.bfloat16):
        raise ValueError("unexpected dense table or ID layout")
    # cat keeps one autograd edge per original parameter. Its backward slices
    # the dense embedding gradient along the same table boundaries.
    weight = torch.cat(weights, dim=0)
    local_ids = ids.movedim(2, 0)
    global_ids = torch.where((local_ids >= 0) & (local_ids < rows),
                             local_ids + offsets, tables * rows).contiguous()
    means = F.embedding(global_ids, weight).mean(dim=-2)
    return OrderedTableSum.apply(means)


# Ordinary eager ATen operations supply the backward of lookup, mean and cat.
# The namespace is experimental; no installed torch_npu package is replaced.
_library = torch.library.Library("npu_hstu_experiment", "DEF")
_library.define("grouped_side_sum(Tensor[] weights, Tensor ids, Tensor offsets) -> Tensor")
_library.impl("grouped_side_sum", grouped_side_sum, "CompositeImplicitAutograd")


def model_side_sum(model, ids):
    model._side_calls += 1
    weights = [table.weight for table in model.side_embeddings]
    first = weights[0]
    config = model.config
    if (type(model.side_embeddings) is not torch.nn.ModuleList or ids.device != first.device
            or first.device.type not in ("npu", "cpu") or first.dtype != torch.bfloat16
            or config.features_per_table != 4 or config.embedding_dim != 64
            or model._side_offsets.device != ids.device
            or any(type(m) is not torch.nn.Embedding or m.padding_idx is not None or m.sparse
                   or m.max_norm is not None or m.scale_grad_by_freq or m._forward_hooks or m._forward_pre_hooks
                   for m in model.side_embeddings)
            or any(w.shape != (config.table_rows, 64) or w.dtype != first.dtype or w.device != first.device
                   or not w.is_contiguous() for w in weights)):
        return None
    result = torch.ops.npu_hstu_experiment.grouped_side_sum(weights, ids, model._side_offsets)
    model._side_hits += 1
    return result
