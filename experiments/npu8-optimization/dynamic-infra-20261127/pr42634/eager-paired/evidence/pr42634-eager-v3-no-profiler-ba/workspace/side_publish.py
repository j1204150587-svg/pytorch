"""Publish one dense side gradient and the ordinary HSTU gradients to the existing 07 pack."""
import os

import torch
from partition_pack import storage_key,stream_for


class OwnedViews(list):
    """Detect any structural change to the private partition list."""
    generation = 0


def mutator(name):
    def changed(self,*args,**kwargs):
        self.generation += 1
        return getattr(list,name)(self,*args,**kwargs)
    return changed


for _name in ("__setitem__","__delitem__","__iadd__","__imul__","append","extend","insert","pop",
              "clear","remove","reverse","sort"):
    setattr(OwnedViews,_name,mutator(_name))


def native_tensor(tensor,shape,pack):
    return (type(tensor) is torch.Tensor and tensor.shape==shape and tensor.device==pack.device
            and tensor.dtype==pack.dtype and tensor.is_contiguous() and tensor.layout==torch.strided
            and not tensor.is_neg() and not tensor.is_conj() and not tensor.requires_grad
            and storage_key(tensor)!=pack.key)


class Publisher:
    def __init__(self,state,pack):
        self.state,self.pack,self.zero=state,pack,state.zero
        self.stats=state.stats
        start=state.stats["group_table_start"]
        end=start+len(state.weights)
        group=state.group
        if (len(group)!=1052 or len(pack.views)!=len(group) or
                tuple(id(p) for p in group[start:end])!=tuple(id(p) for p in state.weights) or
                tuple(p.shape for p in group)!=pack.shapes or pack.flat.device!=state.leaf.device or
                pack.flat.dtype!=state.leaf.dtype or pack.stream!=stream_for(pack.device)):
            raise ValueError("pack_group")
        self.before=tuple(group[:start])
        self.after=tuple(group[end:])
        self.ordinary=self.before+self.after
        self.ordinary_shapes=pack.shapes[:start]+pack.shapes[end:]
        self.slab_shape=(len(state.weights)*state.model.config.table_rows,state.model.config.embedding_dim)
        root=pack.flat._base if pack.flat._base is not None else pack.flat
        offset=pack.flat.storage_offset()
        for view,shape,n in zip(pack.views,pack.shapes,pack.sizes,strict=True):
            if view._base is not root or view.shape!=shape or view.storage_offset()!=offset or not view.is_contiguous():
                raise ValueError("destination_views")
            offset+=n
        self.views=OwnedViews(pack.views)
        pack.views=self.views
        self.ready=None
        self.valid_version=None
        self.stats["direct_installed"]=1

    def prepare(self):
        s,z,p=self.state,self.zero,self.pack
        self.valid_version=None
        if not s.pending:
            self.stats["direct_fallback_no_pending_slab"]+=1
            return False
        if (self.ready is not None or p.pending or not z.is_gradient_accumulation_boundary
                or z._epilogue_ran_this_backward or z.averaged_gradients.get(0) is not None
                or z._grad_acc_hooks or z.params_in_partition[0] is not s.group
                or getattr(z,"_eager_gradient_pack",None) is not p or p.views is not self.views
                or self.views.generation!=0 or p.stream!=stream_for(p.device)
                or z.communication_data_type not in (torch.bfloat16,torch.float32)):
            self.stats["direct_fallback_ownership"]+=1
            return False
        side=s.leaf.grad
        if (not native_tensor(side,self.slab_shape,p) or
                (side.device.type=="npu" and torch.ops.npu.get_npu_format(side)!=2)):
            self.stats["direct_fallback_side_layout"]+=1
            return False
        grads=[param.grad for param in self.ordinary]
        if not all(native_tensor(g,shape,p) for g,shape in zip(grads,self.ordinary_shapes)):
            self.stats["direct_fallback_ordinary_layout"]+=1
            return False
        prefix=len(self.before)
        self.ready=([g.view(-1) for g in grads[:prefix]]+[side.view(-1)]+[g.view(-1) for g in grads[prefix:]])
        return True

    def commit(self):
        if self.ready is None:
            raise RuntimeError("no prepared side interval")
        p,s=self.pack,self.state
        p.counts["calls"]+=1
        result=p.commit_pieces(self.ready,len(s.weights)-1)
        self.ready=None
        s.leaf.grad=None
        s.pending=False
        self.valid_version=p.flat._version
        self.stats["dense_gradients"]+=1
        self.stats["direct_partitions"]+=1
        self.stats["direct_final_views"]+=len(self.views)
        self.stats["ordinary_gradients_checked"]+=len(self.ordinary)
        return result

    def owns(self,shards,flat):
        return (shards is self.views and flat is self.pack.flat and self.views.generation==0
                and self.valid_version is not None and self.valid_version==flat._version)


def install(engine):
    m,z=engine.module,engine.optimizer
    stats=m._side_slab_stats
    if os.getenv("NPU_DYNAMIC_DIRECT_SIDE")!="1":
        stats["direct_disabled"]=1
        return
    state=getattr(m,"_side_slab_owner",None)
    pack=getattr(z,"_eager_gradient_pack",None)
    if state is None or pack is None or not hasattr(pack,"commit_pieces"):
        stats["direct_fallback_install_prerequisites"]=1
        return
    try:
        z._direct_side_publisher=Publisher(state,pack)
    except ValueError as error:
        stats["direct_fallback_install_"+str(error)]=1
