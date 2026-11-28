"""Replay HSTU tensor work while leaving external ZeRO gradient hooks eager."""

import json
import os
from types import MethodType
from pathlib import Path

import torch


def install_single_rank_partition(engine, device):
    from deepspeed import comm as dist
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    zero = engine.optimizer
    groups = zero.params_in_partition
    valid = (type(zero) is DeepSpeedZeroOptimizer and engine.world_size == 1
             and engine.gradient_accumulation_steps() == 1 and zero.partition_gradients
             and not zero.cpu_offload and not zero.has_moe_layers and not zero.zenflow
             and zero.model_parallel_group is None and zero.sequence_parallel_size == 1
             and zero.reduce_scatter and zero.contiguous_gradients and zero.check_grad_overflow
             and zero.dtype == zero.gradient_accumulation_dtype == torch.bfloat16
             and not zero.use_grad_accum_attribute and zero.torch_autocast_gradscaler is None
             and dist.get_world_size(zero.dp_process_group) == 1
             and all(dist.get_world_size(group) == 1 for group in zero.real_dp_process_group)
             and type(zero.optimizer) is torch.optim.AdamW and not zero.round_robin_gradients)
    if not valid or len(zero._grad_acc_hooks) != sum(len(group) for group in groups):
        raise RuntimeError("single-rank ZeRO partition specialization guard failed")
    partitions = []
    for index, group in enumerate(groups):
        shapes = [parameter.shape for parameter in group]
        sizes = [parameter.numel() for parameter in group]
        padding = int(zero.partition_size[index]) - sum(sizes)
        if (tuple(map(id, group)) != tuple(map(id, zero.bit16_groups[index]))
                or zero.first_offset[index] != 0 or padding != zero.groups_padding[index]
                or padding < 0 or any(getattr(parameter, "use_muon", False) for parameter in group)):
            raise RuntimeError("unexpected single-rank ZeRO partition topology")
        partitions.append((group, shapes, sizes, torch.zeros(padding, dtype=zero.dtype, device=device) if padding else None))
    for hook in zero._grad_acc_hooks:
        hook.remove()
    zero._grad_acc_hooks.clear()
    zero._recsys_local_flat_gradients = {}
    counts = {"partition": 0}
    optimized = None
    if os.getenv("NPU_DYNAMIC_PARTITION_PACK")=="1":
        from partition_pack import GradientPack
        if len(partitions)!=1 or partitions[0][3] is not None or zero.communication_data_type not in (torch.bfloat16, torch.float32):
            raise RuntimeError("gradient pack requires one unpadded unscaled BF16 partition")
        _,shapes,sizes,_ = partitions[0]
        optimized = GradientPack(shapes,sizes,device,zero.dtype)
        zero._eager_gradient_pack = optimized

    @torch.no_grad()
    def partition(self):
        if not self.is_gradient_accumulation_boundary or self._epilogue_ran_this_backward:
            raise RuntimeError("single-rank partition expects one backward at an accumulation boundary")
        for group_index, (group, shapes, sizes, padding) in enumerate(partitions):
            direct=getattr(self,"_direct_side_publisher",None)
            if direct is not None and direct.ready is not None:
                if group_index!=0 or group is not direct.state.group:
                    raise RuntimeError("side gradient delivered to wrong partition")
                views,flattened=direct.commit()
                self._recsys_local_flat_gradients[group_index]=(views,flattened)
                self.averaged_gradients[group_index]=views
                self.all_grad_tensors[group_index]=None
                self.free_grad_in_param_list(direct.ordinary)
                continue
            gradients = [parameter.grad for parameter in group]
            if any(grad is None or grad.dtype != self.dtype or grad.device != device or not grad.is_contiguous()
                   for grad in gradients):
                raise RuntimeError("unexpected gradient in single-rank ZeRO partition")
            packed = None
            if optimized is not None:
                if self.averaged_gradients.get(group_index) is not None:
                    raise RuntimeError("previous optimizer partition has not been cleared")
                packed = optimized.pack(gradients)
            if packed is not None:
                views,flattened = packed
            else:
                flattened = torch.cat([grad.view(-1) for grad in gradients])
                if self.communication_data_type not in (torch.bfloat16, torch.float32):
                    flattened = flattened.to(self.communication_data_type).to(self.dtype)
                views = [part.view(shape) for part, shape in zip(flattened.split(sizes), shapes, strict=True)]
                if padding is not None:
                    views.append(padding.clone())
            self._recsys_local_flat_gradients[group_index] = (views, flattened)
            self.averaged_gradients[group_index] = views
            self.all_grad_tensors[group_index] = None
            self.free_grad_in_param_list(group)
        self._epilogue_ran_this_backward = True
        counts["partition"] += 1

    zero.overlapping_partition_gradients_reduce_epilogue = MethodType(partition, zero)
    return counts


def install(engine, config, batch_size, device, rank_dir):
    from model import MultiTableHSTU

    original = engine.module
    live = dict(original.named_parameters())
    if engine.world_size != 1 or engine.gradient_accumulation_steps() != 1 or config.dropout != 0:
        raise RuntimeError("HSTU graph requires one rank, accumulation 1 and no dropout")
    if any(weight.dtype != torch.bfloat16 or weight.device != device or not weight.requires_grad
           for weight in live.values()):
        raise RuntimeError("unexpected live HSTU parameter placement or dtype")

    with torch.random.fork_rng(devices=[]):
        shadow = MultiTableHSTU(config)
    detached = {name: torch.nn.Parameter(weight.detach(), requires_grad=True) for name, weight in live.items()}
    for name, weight in detached.items():
        parent, _, leaf = name.rpartition(".")
        setattr(shadow.get_submodule(parent), leaf, weight)
    shadow.output.weight = shadow.item_embedding.weight
    if tuple(dict(shadow.named_parameters())) != tuple(live):
        raise RuntimeError("shadow HSTU parameter topology does not match the original")
    aliases = tuple(detached.values())
    parameters = tuple(live.values())
    pointers = tuple(weight.data_ptr() for weight in parameters)
    if pointers != tuple(weight.data_ptr() for weight in aliases):
        raise RuntimeError("graph parameters must alias ZeRO's live BF16 weights")

    shape = (batch_size, config.seq_len)
    static_args = (torch.ones(shape, dtype=torch.long, device=device),
                   torch.zeros(batch_size, dtype=torch.long, device=device),
                   torch.zeros((*shape, config.num_tables, config.features_per_table), dtype=torch.long, device=device))
    stream = torch.npu.Stream()
    torch.npu.synchronize()
    with torch.npu.stream(stream):
        for _ in range(3):
            output = shadow(*static_args)
            torch.autograd.grad(output, aliases, grad_outputs=torch.ones_like(output))
    torch.npu.synchronize()
    pool = torch.npu.graph_pool_handle()
    forward_graph, backward_graph = torch.npu.NPUGraph(), torch.npu.NPUGraph()
    with torch.npu.graph(forward_graph, pool=pool, stream=stream):
        static_output = shadow(*static_args)
    static_gradient = torch.empty_like(static_output)
    with torch.npu.graph(backward_graph, pool=pool, stream=stream):
        static_gradients = torch.autograd.grad(static_output, aliases, grad_outputs=static_gradient)
    torch.npu.synchronize()

    replay_counts = {"forward": 0, "backward": 0}

    class Replay(torch.autograd.Function):
        @staticmethod
        def forward(context, items, groups, features, *weights):
            for target, source in zip(static_args, (items, groups, features), strict=True):
                if target.shape != source.shape or target.device != source.device or target.dtype != source.dtype:
                    raise RuntimeError("dynamic input incompatible with captured training graph")
                target.copy_(source)
            forward_graph.replay()
            replay_counts["forward"] += 1
            return static_output.detach()

        @staticmethod
        @torch.autograd.function.once_differentiable
        def backward(context, gradient):
            static_gradient.copy_(gradient)
            backward_graph.replay()
            replay_counts["backward"] += 1
            return (None, None, None, *(value.detach() for value in static_gradients))

    def graphed_forward(items, groups, features):
        if not original.training or any(parameter.data_ptr() != pointer
                                        for parameter, pointer in zip(parameters, pointers, strict=True)):
            raise RuntimeError("captured weights moved or model left training mode")
        return Replay.apply(items, groups, features, *parameters)

    def finish(expected):
        record = {"parameter_aliases": len(pointers), "counts": replay_counts, "expected_steps": expected}
        (Path(rank_dir) / "hstu_graph.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        if any(value != expected for value in replay_counts.values()):
            raise RuntimeError(f"incomplete training replay: {record}")

    original.forward = graphed_forward
    return finish
