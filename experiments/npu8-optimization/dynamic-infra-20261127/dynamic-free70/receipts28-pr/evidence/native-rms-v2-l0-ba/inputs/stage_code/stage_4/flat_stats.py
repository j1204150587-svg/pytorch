"""Eager overflow and row-preserving norm over the BF16 buffer published by 07."""
import json
import os
from pathlib import Path

import torch


def nonfinite(flat):
    # isfinite tests elements independently; a sum of several large finite
    # numbers may overflow and must not cause DeepSpeed to skip an update.
    return torch.isfinite(flat).all().logical_not().reshape(1)


def norm_plan(group, stage, zero, max_elements, max_tensors):
    plan = []
    offset = 0
    for param in group:
        shape = tuple(param.shape)
        width = param.numel()
        if (width == 0 or getattr(param,stage.PIPE_REPLICATED,False) or
                not (stage.is_model_parallel_parameter(param) or zero.model_parallel_rank==0)):
            return None
        if (not plan or plan[-1][1]!=shape or plan[-1][2]>=max_tensors or
                (plan[-1][2]+1)*width>max_elements):
            plan.append([offset,shape,1,width])
        else:
            plan[-1][2] += 1
        offset += width
    return plan


def row_norm_square(flat,plan):
    values = []
    for offset,shape,count,width in plan:
        chunk = flat.narrow(0,offset,count*width)
        if count==1:
            # Preserve the singleton shape presented to the baseline norm.
            value = torch.linalg.vector_norm(chunk.view(shape).double().detach(),ord=2).reshape(1)
        else:
            value = torch.linalg.vector_norm(chunk.view(count,width).double().detach(),ord=2,dim=1)
        values.append(value)
    norms = values[0] if len(values)==1 else torch.cat(values)
    return norms.square().sum().float()


def install(zero,norm_stats,rank_dir):
    mode = os.environ.get("RECSYS_DYNAMIC_FLAT_STATS","off")
    if mode == "off":
        return
    if mode not in ("overflow","norm","both"):
        raise ValueError(mode)
    from deepspeed.runtime.zero import stage_1_and_2 as stage
    from overflow import shard_metadata
    from zero2_batch_norm import CHUNK_ELEMENTS,CHUNK_TENSORS
    if (not hasattr(zero,"_recsys_local_flat_gradients") or norm_stats is None or
            len(zero.params_in_partition)!=1 or zero.groups_padding!=[0] or
            stage.dist.get_world_size()!=1 or zero.dtype!=torch.bfloat16 or zero.cpu_offload):
        raise ValueError("flat stats requires dynamic 07 single rank unpadded BF16 partition and batched norm")
    group, = zero.params_in_partition
    fmt = torch.ops.npu.get_npu_format
    device = torch.device(zero.device)
    plan = norm_plan(group,stage,zero,CHUNK_ELEMENTS,CHUNK_TENSORS)
    if plan is None:
        raise ValueError("unsupported norm layout")
    shapes = [p.shape for p in group]
    sizes = [p.numel() for p in group]
    total = sum(sizes)
    records = dict(mode=mode,overflow=0,norm=0,overflow_fallback=0,norm_fallback=0)
    packed = [(n,w) for _,_,n,w in plan if n>1]

    def eligible(shards):
        current = zero._recsys_local_flat_gradients.get(0)
        if current is None or current[0] is not shards or len(shards)!=len(shapes):
            return None
        flat = current[1]
        if (type(flat) is not torch.Tensor or flat.device!=device or flat.dtype!=torch.bfloat16 or
                flat.shape!=(total,) or not flat.is_contiguous() or flat.requires_grad or fmt(flat)!=2):
            return None
        # Validate actual layout, including replacement/reordering inside the
        # same list. _base identifies the shared storage without a device call.
        root = flat._base if flat._base is not None else flat
        offset = flat.storage_offset()
        for grad,shape,size in zip(shards,shapes,sizes,strict=True):
            if (type(grad) is not torch.Tensor or grad._base is not root or
                    grad.device!=device or grad.dtype!=flat.dtype or grad.requires_grad or
                    grad.is_neg() or grad.is_conj() or grad.shape!=shape or
                    not grad.is_contiguous() or grad.storage_offset()!=offset):
                return None
            offset += size
        return flat

    if mode in ("overflow","both"):
        legacy = zero.has_overflow_partitioned_grads_serial
        scan = zero._recsys_overflow_stats
        def local_scan():
            shards = zero.averaged_gradients[0]
            flat = eligible(shards)
            if flat is None:
                records["overflow_fallback"]+=1
                return legacy()
            if scan.first_shards is None:
                scan.first_shards=shard_metadata([shards])
                scan.first_npu_formats={2:len(shards)}
            invalid = nonfinite(flat)
            scan.calls+=1
            scan.fast_calls+=1
            scan.chunks+=1
            records["overflow"]+=1
            return invalid
        zero.has_overflow_partitioned_grads_serial=local_scan

    if mode in ("norm","both"):
        original_norm=zero.get_grad_norm_direct
        def group_norm(gradients,params,norm_type=2):
            flat=eligible(gradients) if params is group and norm_type==2 else None
            if flat is None:
                records["norm_fallback"]+=1
                return original_norm(gradients,params,norm_type)
            result=row_norm_square(flat,plan)
            stage.dist.all_reduce(result,op=stage.dist.ReduceOp.SUM,group=zero.dp_process_group)
            zero._model_parallel_all_reduce(tensor=result,op=stage.dist.ReduceOp.SUM)
            result=result.pow(0.5)
            stage.mask_nan_or_inf_with_val_inplace(result,device=zero.device)
            norm_stats.calls+=1
            norm_stats.batched_calls+=1
            norm_stats.batches+=len(packed)
            norm_stats.batched_gradients+=sum(n for n,_ in packed)
            records["norm"]+=1
            return result
        zero.get_grad_norm_direct=group_norm

    import atexit
    def save():
        (Path(rank_dir)/"flat_stats_counts.json").write_text(json.dumps(records,indent=2))
    atexit.register(save)
