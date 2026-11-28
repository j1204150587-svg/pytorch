"""Expose a completed ZeRO IPG bucket as one range copy to the existing PTA copy dispatcher."""
import ast
import hashlib
import inspect
import json
import os
from pathlib import Path
import textwrap
from types import MethodType
import torch


SOURCE_SHA = {
    "get_gradient_for_reduction":"c04cda358ae7b18f229bc9f65f2218d92a5d58b5ada7105dc588ac3978e05c0a",
    "copy_grads_in_partition":"6a272edd5f87a80ce0506708c79cd73e2ad772c1f78f69118a885653cc0de673",
    "reduce_ipg_grads":"64123ab6c86b21675027b8055500238b4344d30a4d819bf10f3256d508091d08",
}


def pinned(method,name):
    text=textwrap.dedent(inspect.getsource(method))
    node=ast.parse(text).body[0]
    body="\n".join(text.splitlines()[node.lineno-1:node.end_lineno]).strip()
    if hashlib.sha256(body.encode()).hexdigest()!=SOURCE_SHA[name]:
        raise ValueError("unrecognized DeepSpeed source "+name)
    return body


def nd(t):
    return t.device.type=="cpu" or torch.ops.npu.get_npu_format(t)==2


def storage_id(t):
    return t.untyped_storage()._cdata


class BucketCopy:
    def __init__(self,zero,mode):
        self.z=zero
        self.mode=mode
        self.attempts=[]
        self.hits=0
        self.declined=0
        self.source_verified=False

    def prepare(self,bucket):
        z=self.z
        if (not z.partition_gradients or not z.contiguous_gradients or z.cpu_offload or z.zenflow
                or z.use_grad_accum_attribute or len(z.params_in_partition)!=1 or len(z.bit16_groups)!=1
                or len(bucket.params)<2 or len(bucket.params)!=len(z.params_in_partition[0])
                or z.dtype!=torch.bfloat16 or z.grads_in_partition is not None
                or z.round_robin_gradients or id(z.params_in_partition[0])!=self.group_id):
            return "optimizer_contract",None
        if not z.overlap_comm or (bucket.buffer is None or bucket.buffer[bucket.index] is None):
            return "ipg_buffer",None
        ipg=bucket.buffer[bucket.index]
        if (ipg.ndim!=1 or not ipg.is_contiguous() or ipg.dtype!=z.dtype or not nd(ipg)
                or ipg.is_conj() or ipg.is_neg() or ipg.requires_grad or ipg.device.type not in ("npu","cpu")):
            return "ipg_layout",None
        if ipg.device.type=="npu" and torch.npu.current_stream(ipg.device)!=z.reduction_stream:
            return "stream",None
        elements=int(bucket.elements)
        if not 0<elements<=ipg.numel() or elements!=self.group_elements:
            return "ipg_extent",None
        source=storage_id(ipg)
        cursor=ipg.storage_offset()
        views=[]
        ids=set()
        breaks=0
        expected=cursor
        for group_idx,index,param_id in bucket.params:
            if (group_idx!=0 or z.params_already_reduced[param_id] or not z.is_param_in_current_partition[param_id]
                    or param_id in ids):
                return "partition_flags",None
            p=z.bit16_groups[group_idx][index]
            if z.get_param_id(p)!=param_id:
                return "parameter_id",None
            g=z.get_gradient_for_reduction(p)
            if (type(g) is not torch.Tensor or g.requires_grad or g.is_conj() or g.is_neg()
                    or g.device!=ipg.device or g.dtype!=ipg.dtype or not g.is_contiguous()
                    or g.shape!=p.shape or not g.numel()):
                return "grad_layout",None
            if storage_id(g)!=source:
                return "different_ipg_storage",None
            if g.storage_offset()!=expected:
                breaks+=1
            expected=g.storage_offset()+g.numel()
            views.append((param_id,g,tuple(g.shape),g.numel()))
            ids.add(param_id)
        if breaks or expected!=cursor+elements:
            return f"nonadjacent:{breaks}",None
        return None,(ipg,elements,views)

    def try_bucket(self,bucket):
        if self.mode=="off" or not bucket.params:
            return False
        reason,payload=self.prepare(bucket)
        self.attempts.append({"n":len(bucket.params), "elements":int(bucket.elements), "reason":reason})
        if reason:
            self.declined+=1
            return False
        if self.mode=="observe":
            return False
        ipg,n,views=payload
        z=self.z
        # Allocate after all fallbacks. Source belongs to the still-live IPG bucket.
        # The new destination is isolated and cannot overlap any source member.
        dst=torch.empty(n,dtype=ipg.dtype,device=ipg.device)
        source=ipg.narrow(0,0,n)
        prepared=[]
        cursor=0
        for param_id,g,shape,count in views:
            prepared.append((param_id,g,dst.narrow(0,cursor,count).view(shape)))
            cursor+=count
        if ipg.device.type=="npu":
            stream=torch.npu.current_stream(ipg.device)
            ipg.record_stream(stream)
            dst.record_stream(stream)
        # One regular eager PTA copy. After this point an error propagates; no replay.
        with torch.profiler.record_function("NPUIPGBucketRangeCopy"):
            dst.copy_(source)
        # The source method modifies a different view of dst for each parameter.
        # All of those views share dst's autograd version counter.
        torch.autograd.graph.increment_version([dst]*(len(prepared)-1))
        z.grads_in_partition=dst
        for param_id,g,v in prepared:
            z.params_already_reduced[param_id]=True
            g.data=v.data
        z.grads_in_partition_offset=n
        self.hits+=1
        return True

    def finish(self,rank):
        (Path(rank)/"ipg_range_copy.json").write_text(json.dumps(
            dict(mode=self.mode, hits=self.hits, declined=self.declined, attempts=self.attempts,
                 source_verified=self.source_verified),indent=2)+"\n")


def install(engine):
    z=engine.optimizer
    mode=os.environ.get("NPU_DYNAMIC_IPG_RANGE_COPY","off")
    if mode not in ("off","observe","range"): raise ValueError(mode)
    state=BucketCopy(z,mode)
    if engine.world_size!=1 or engine.gradient_accumulation_steps()!=1:
        return state
    texts={}
    for name in SOURCE_SHA:
        if name in vars(z): raise ValueError("instance has an existing override: "+name)
        texts[name]=pinned(getattr(z,name),name)
    state.source_verified=True
    if mode=="off": return state
    state.group_id=id(z.params_in_partition[0])
    state.group_elements=sum(p.numel() for p in z.params_in_partition[0])
    old=z.reduce_ipg_grads
    anchor='            for group_idx, param_idx_in_group, param_id in bucket.params:'
    src=texts["reduce_ipg_grads"]
    if src.count(anchor)!=1: raise ValueError("unrecognized bucket iteration")
    replacement=('            if self._try_ipg_range_copy(bucket):\n'
                 '                bucket.clear()\n                continue\n'+anchor)
    out={}
    exec(compile(src.replace(anchor,replacement),"<verified DS reduce_ipg_grads with bucket copy>","exec"),
         old.__func__.__globals__,out)
    z._try_ipg_range_copy=state.try_bucket
    z.reduce_ipg_grads=MethodType(out["reduce_ipg_grads"],z)
    return state
