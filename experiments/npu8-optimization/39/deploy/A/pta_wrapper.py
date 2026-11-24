"""Exercise the PTA single-buffer graph wrapper for local ZeRO statistics."""

import json
import os
from pathlib import Path
from types import MethodType

import torch


def _wrapper():
    from torch_npu.npu import graphs

    source = Path(__file__).with_name("pta_graph_function.py")
    exec(compile(source.read_text(encoding="utf-8"), str(source), "exec"), vars(graphs))
    return graphs._make_graphed_buffer_callable


def install(engine, rank_dir, norm_stats):
    import deepspeed.runtime.zero.stage_1_and_2 as stage
    from overflow import ACL_FORMAT_ND
    from zero2_batch_norm import _plan_batches, CHUNK_ELEMENTS, CHUNK_TENSORS

    zero = engine.optimizer
    group, = zero.params_in_partition
    padding, = zero.groups_padding
    if padding or engine.world_size != 1 or zero.model_parallel_group is not None:
        raise RuntimeError("PTA wrapper experiment expects one full, unpadded parameter group")
    shape = [weight.shape for weight in group]
    sizes = [weight.numel() for weight in group]
    device = torch.device(zero.device)
    source = torch.zeros(sum(sizes), device=device, dtype=torch.bfloat16)
    example = [shard.view(size) for shard, size in zip(source.split(sizes), shape, strict=True)]
    batch_plan, reason = _plan_batches(zero, example, group, stage, device, torch.ops.npu.get_npu_format,
                                      CHUNK_ELEMENTS, CHUNK_TENSORS)
    if reason is not None:
        raise RuntimeError(reason)
    if sum(grad.numel() for batch in batch_plan for grad in batch) != source.numel():
        raise RuntimeError("norm batch plan must cover the entire flat group in order")
    layout = [(len(batch), batch[0].numel()) for batch in batch_plan]
    batched_blocks = sum(count > 1 for count, _ in layout)
    batched_gradients = sum(count for count, _ in layout if count > 1)
    graph_mode = os.environ.get("RECSYS_PTA_REPLAY", "1") == "1"
    fast = _wrapper() if graph_mode else None
    scope = os.environ.get("RECSYS_PTA_SCOPE", "both")

    def norm_square(flat):
        norms = []
        offset = 0
        for count, width in layout:
            chunk = flat.narrow(0, offset, count * width)
            offset += count * width
            if count == 1:
                norms.append(torch.linalg.vector_norm(chunk.double(), ord=2).reshape(1))
            else:
                norms.append(torch.linalg.vector_norm(chunk.view(count, width).double(), ord=2, dim=1))
        return torch.cat(norms).square().sum().float()

    def nonfinite(flat):
        return zero._has_inf_or_nan(flat).bool().reshape(1)

    if scope not in ("norm", "overflow", "both"):
        raise ValueError(scope)

    def wrap(function):
        if graph_mode:
            return fast(function, source)
        static = torch.empty_like(source)

        def eager(value):
            static.copy_(value)
            return function(static)

        return eager

    run_norm = wrap(norm_square) if scope in ("norm", "both") else None
    run_overflow = wrap(nonfinite) if scope in ("overflow", "both") else None
    counts = dict(scope=scope, graph=graph_mode, norm=0, overflow=0)
    def flat(shards):
        current_shards, tensor = zero._recsys_local_flat_gradients[0]
        if (shards is not current_shards or tensor.shape != source.shape or tensor.dtype != source.dtype
                or tensor.device != device or not tensor.is_contiguous()
                or torch.ops.npu.get_npu_format(tensor) != ACL_FORMAT_ND):
            raise RuntimeError("gradient group does not match staged layout")
        return tensor

    if run_norm is not None:
        def get_norm(self, gradients, params, norm_type=2):
            if params is not group or norm_type != 2:
                raise RuntimeError("unexpected norm group")
            total = run_norm(flat(gradients))
            stage.dist.all_reduce(total, op=stage.dist.ReduceOp.SUM, group=self.dp_process_group)
            self._model_parallel_all_reduce(tensor=total, op=stage.dist.ReduceOp.SUM)
            total = total.pow(0.5)
            stage.mask_nan_or_inf_with_val_inplace(total, device=self.device)
            counts["norm"] += 1
            norm_stats.calls += 1
            norm_stats.batched_calls += 1
            norm_stats.batches += batched_blocks
            norm_stats.batched_gradients += batched_gradients
            return total

        zero.get_grad_norm_direct = MethodType(get_norm, zero)

    if run_overflow is not None:
        from overflow import shard_metadata

        scan = zero._recsys_overflow_stats

        def check_overflow(self):
            group_gradients = self.averaged_gradients[0]
            if scan.first_shards is None:
                scan.first_shards = shard_metadata([group_gradients])
                scan.first_npu_formats = {ACL_FORMAT_ND: len(group_gradients)}
            result = run_overflow(flat(group_gradients))
            scan.calls += 1
            scan.fast_calls += 1
            scan.chunks += 1
            counts["overflow"] += 1
            return result

        zero.has_overflow_partitioned_grads_serial = MethodType(check_overflow, zero)

    local_step_mode = os.environ.get("RECSYS_ZERO2_LOCAL_STEP")
    finish_local_step = None
    if local_step_mode:
        from local_step import install as install_local_step

        finish_local_step = install_local_step(zero, device, local_step_mode, rank_dir)

    def finish():
        if finish_local_step is not None:
            finish_local_step()
        (Path(rank_dir) / "pta_wrapper.json").write_text(json.dumps(counts, indent=2) + "\n", encoding="utf-8")

    return finish
