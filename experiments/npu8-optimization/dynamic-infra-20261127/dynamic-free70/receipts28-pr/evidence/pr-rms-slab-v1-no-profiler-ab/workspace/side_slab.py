"""One eager dense-gradient leaf for the guarded contiguous ZeRO-2 side-table interval."""
from collections import Counter
import json
import os
from pathlib import Path
from types import MethodType

import torch
from torch.nn.modules import module as module_hooks


def has_hooks(table, param):
    return bool(table._forward_hooks or table._forward_pre_hooks or table._backward_hooks
                or table._backward_pre_hooks or param._backward_hooks or param._post_accumulate_grad_hooks)


def python_parameter_guard(live,tables,weights,pointers):
    return not (len(live) != len(tables) or any(
                current is not prior or current.weight is not w or w.data_ptr() != ptr or not w.requires_grad
                or w.grad is not None or has_hooks(current,w)
                for current,prior,w,ptr in zip(live,tables,weights,pointers)))


def global_hooks():
    return bool(module_hooks._global_forward_hooks or module_hooks._global_forward_pre_hooks
                or module_hooks._global_backward_hooks or module_hooks._global_backward_pre_hooks)


class SideSlab:
    def __init__(self, model, zero, stats):
        self.model, self.zero, self.stats = model, zero, stats
        self.group = zero.params_in_partition[0]
        self.flat = zero.bit16_groups_flat[0]
        self.first_ptr = self.flat.data_ptr()
        self.tables = tuple(model.side_embeddings)
        from hstu_slab_guard import SlabParameterGuard
        self.parameter_guard = SlabParameterGuard(stats)
        self.weights = tuple(t.weight for t in self.tables)
        t, r, d = len(self.weights), model.config.table_rows, model.config.embedding_dim
        if (t != 1024 or r != 64 or d != 64 or model.config.features_per_table != 4
                or self.flat.ndim != 1 or self.flat.dtype != torch.bfloat16 or not self.flat.is_contiguous()
                or len({id(w) for w in self.weights}) != t):
            raise ValueError("model_layout")
        if global_hooks() or any(type(m) is not torch.nn.Embedding or m.max_norm is not None or m.sparse
                                 or m.scale_grad_by_freq or m.padding_idx is not None or has_hooks(m,w)
                                 for m,w in zip(self.tables,self.weights)):
            raise ValueError("embedding_contract_or_hooks")
        group_order = [id(p) for p in self.group]
        group_start = group_order.index(id(self.weights[0])) if id(self.weights[0]) in group_order else -1
        if group_start < 0 or group_order[group_start:group_start+t] != [id(w) for w in self.weights]:
            raise ValueError("group_order")
        self.offset = sum(p.numel() for p in self.group[:group_start])
        self.pointers = tuple(self.first_ptr + (self.offset+i*r*d)*self.flat.element_size() for i in range(t))
        if (self.offset+t*r*d > self.flat.numel() or any(
                not w.requires_grad or not w.is_leaf or w.device != self.flat.device or w.dtype != self.flat.dtype
                or w.shape != (r,d) or not w.is_contiguous() or w.data_ptr() != ptr
                or w.untyped_storage().data_ptr() != self.flat.untyped_storage().data_ptr()
                for w,ptr in zip(self.weights,self.pointers))):
            raise ValueError("flat_address_or_stride")
        self.leaf = self.flat.narrow(0,self.offset,t*r*d).view(t*r,d).detach().requires_grad_(True)
        self.buffer = torch.empty((t,r,d), dtype=self.flat.dtype, device=self.flat.device)
        self.views = self.buffer.unbind(0)
        self.pending = False
        self.stats.update(layout_checks=1, views_created=t, group_table_start=group_start, flat_element_offset=self.offset)

    def forward(self, ids):
        if self.pending:
            raise RuntimeError("side-slab requires one matched forward/backward per ZeRO epilogue")
        m,z = self.model,self.zero
        # The single-rank runner is the owner of the model and changes values through
        # this flat. Refuse out-of-band rewiring before using a cached slab address.
        if (not m.training or not torch.is_grad_enabled() or self.leaf.grad is not None
                or z.bit16_groups_flat[0] is not self.flat or self.flat.data_ptr() != self.first_ptr
                or z.params_in_partition[0] is not self.group or z._grad_acc_hooks or global_hooks()
                or ids.device != self.flat.device or m._side_offsets.device != ids.device
                or ids.dtype not in (torch.int32,torch.int64)):
            self.stats["runtime_fallback_environment"] += 1
            return None
        # A single light identity/pointer/hook pass replaces the full v1 guards.
        # It catches table replacement, .data rebinding, registered hooks and frozen weights.
        if not self.parameter_guard.matches(self,m.side_embeddings,python_parameter_guard):
            self.stats["runtime_fallback_parameter"] += 1
            return None
        result = torch.ops.npu_hstu_experiment.slab_side_sum(self.leaf,ids,m._side_offsets,m.config.table_rows)
        self.pending = True
        self.stats["slab_forwards"] += 1
        return result

    @torch.no_grad()
    def publish(self):
        if not self.pending:
            return
        grad = self.leaf.grad
        if (grad is None or grad.dtype != self.flat.dtype or grad.shape != self.leaf.shape
                or grad.device != self.flat.device or not grad.is_contiguous()
                or any(w.grad is not None for w in self.weights)):
            raise RuntimeError("side slab has missing, mixed or incompatible gradients")
        # One real copy per step keeps 1,024 cached views valid across arbitrary
        # allocator addresses. The ordinary 07 subsequently owns its output.
        self.buffer.view_as(grad).copy_(grad)
        for w,view in zip(self.weights,self.views):
            w.grad = view
        self.leaf.grad = None
        self.pending = False
        self.stats["dense_gradients"] += 1
        self.stats["gradient_buffer_copies"] += 1
        self.stats["published_table_views"] += len(self.views)


def install(engine):
    model,zero = engine.module,engine.optimizer
    stats = Counter()
    model._side_slab_stats = stats
    if os.environ.get("NPU_DYNAMIC_SIDE_SLAB") != "1":
        stats["disabled"] = 1
        return
    if (not hasattr(model,"_side_offsets") or not hasattr(zero,"_recsys_local_flat_gradients")
            or zero._grad_acc_hooks or engine.world_size != 1 or engine.gradient_accumulation_steps() != 1
            or len(zero.params_in_partition) != 1 or tuple(zero.groups_padding) != (0,)
            or zero.first_offset[0] != 0 or len(zero.bit16_groups_flat) != 1):
        stats["install_fallback_prerequisites"] = 1
        return
    try:
        state = SideSlab(model,zero,stats)
    except ValueError as error:
        stats["install_fallback_" + str(error)] = 1
        return
    model._side_slab_owner = state
    stats["installed"] = 1
    original = zero.overlapping_partition_gradients_reduce_epilogue
    def partition(self, *args, **kwargs):
        direct=getattr(self,"_direct_side_publisher",None)
        if direct is None or not direct.prepare():
            state.publish()
        return original(*args, **kwargs)
    zero.overlapping_partition_gradients_reduce_epilogue = MethodType(partition,zero)


def finish(model, rank_dir):
    owner = getattr(model,"_side_slab_owner",None)
    if owner is not None and (owner.pending or owner.leaf.grad is not None):
        raise RuntimeError("unconsumed final side-slab gradient")
    (Path(rank_dir)/"side_slab_counts.json").write_text(json.dumps(dict(model._side_slab_stats),indent=2))
