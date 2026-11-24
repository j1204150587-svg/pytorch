"""Side-table batching for the shadow model used by the 09 replay path."""

from types import MethodType

import torch


TABLES = 1024
PARAMETERS = 1052


def configure_shadow(shadow, optimizer, parameters, batch_size, device):
    from sidepack_npu import side_features

    if not callable(getattr(shadow, "_side_feature_sum", None)):
        raise RuntimeError("apply the model.py hunk of graph-sidepack.patch before capture")
    config = shadow.config
    tables = shadow.side_embeddings
    groups = optimizer.params_in_partition
    if (config.num_tables != TABLES or len(tables) != TABLES or len(parameters) != PARAMETERS
            or len(groups) != 1 or tuple(optimizer.groups_padding) != (0,)
            or len(groups[0]) != PARAMETERS
            or set(map(id, groups[0])) != set(map(id, parameters))):
        raise RuntimeError("graph sidepack requires 1024 independent tables and one unpadded 1052-parameter ZeRO group")
    if (batch_size * config.seq_len * config.embedding_dim) % 128:
        raise RuntimeError("the 05 Triton ordered fold requires B * L * D divisible by 128")
    for index, table in enumerate(tables):
        weight = table.weight
        if (type(table) is not torch.nn.Embedding or table.padding_idx is not None
                or table.max_norm is not None or table.scale_grad_by_freq or table.sparse
                or weight.shape != (config.table_rows, config.embedding_dim)
                or weight.device != device or weight.dtype != torch.bfloat16 or not weight.is_contiguous()):
            raise RuntimeError(f"unsupported graph sidepack embedding contract at table {index}")

    # The shadow's parameters have already been replaced with aliases of the
    # live NPU weights. Move just this buffer before warmup, not the model.
    offsets = torch.arange(TABLES, dtype=torch.long, device="cpu") * config.table_rows
    offsets = offsets.view(TABLES, 1, 1, 1).to(device)
    shadow.register_buffer("_sidepack_offsets", offsets, persistent=False)

    def lookup(self, feature_ids):
        if feature_ids.device != self._sidepack_offsets.device:
            raise RuntimeError("sidepack offsets and feature IDs must be on the same device")
        return side_features(self.side_embeddings, feature_ids, self._sidepack_offsets, config.table_rows)

    shadow._side_feature_sum = MethodType(lookup, shadow)
    return offsets


class GradientLifetime:
    """Guard graph-owned gradient views until 09 has partitioned and stepped."""

    def __init__(self, optimizer, parameters, gradients, stream, offsets):
        self.optimizer = optimizer
        self.stream = stream
        # A graph uses the address of this pre-capture buffer even if the
        # shadow and its Python autograd graph have been collected.
        self.offsets = offsets
        self.generation = 0
        self.backward_done = True
        self.partition_id = None
        self.gradient_storages = set()
        for index, (parameter, gradient) in enumerate(zip(parameters, gradients, strict=True)):
            if (gradient is None or gradient.shape != parameter.shape or gradient.dtype != parameter.dtype
                    or gradient.device != parameter.device or gradient.requires_grad or not gradient.is_contiguous()):
                raise RuntimeError(f"unsupported captured gradient layout at parameter {index}")
            # CatBackward can return disjoint views of one packed allocation.
            self.gradient_storages.add(gradient.untyped_storage().data_ptr())

    def _partitions(self):
        partitions = getattr(self.optimizer, "_recsys_local_flat_gradients", None)
        if partitions is None:
            raise RuntimeError("graph sidepack requires 09's single-rank partition epilogue")
        return partitions

    def require_consumed(self):
        if not self.backward_done:
            raise RuntimeError("previous graph sidepack forward still awaits backward")
        if self.generation:
            published = self._partitions().get(0)
            if (published is None or id(published) == self.partition_id
                    or self.optimizer.averaged_gradients.get(0) is not None):
                raise RuntimeError("previous graph gradients must be partitioned and cleared by engine.step()")
            if published[1].untyped_storage().data_ptr() in self.gradient_storages:
                raise RuntimeError("09 partition must own a gradient buffer independent of replay storage")

    def begin_forward(self, stream):
        if stream != self.stream:
            raise RuntimeError("graph sidepack replay requires the original training stream")
        self.require_consumed()
        self.partition_id = id(self._partitions().get(0))
        self.generation += 1
        self.backward_done = False
        return self.generation

    def require_backward(self, generation, stream):
        if stream != self.stream or generation != self.generation or self.backward_done:
            raise RuntimeError("graph sidepack requires one matching backward on the forward stream")
