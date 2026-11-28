"""Pack adjacent logical gradients sharing storage using one cat and reusable private ZeRO views."""
import torch


def stream_for(device):
    return torch.npu.current_stream(device) if device.type=="npu" else None


def storage_key(tensor):
    return tensor.untyped_storage()._cdata


def segments(gradients,sizes,output_key):
    pieces=[]
    first=None
    current_key=None
    next_offset=length=0
    shared=0

    def finish():
        if length==first.numel():
            return first.view(-1)
        return first.as_strided((length,),(1,),first.storage_offset())

    for grad,numel in zip(gradients,sizes,strict=True):
        key=storage_key(grad)
        if key==output_key:
            return None,0
        offset=grad.storage_offset()
        if key==current_key and offset==next_offset:
            # Format belongs to the NPU storage. Only collapse a plain ND region.
            if length!=first.numel() or grad.device.type!="npu" or torch.ops.npu.get_npu_format(first)==2:
                length+=numel
                next_offset+=numel
                shared+=1
                continue
        if first is not None:
            pieces.append(finish())
        first,current_key,length,next_offset=grad,key,numel,offset+numel
    if first is not None:
        pieces.append(finish())
    return pieces,shared


class GradientPack:
    def __init__(self,shapes,sizes,device,dtype):
        self.shapes=tuple(shapes)
        self.sizes=tuple(sizes)
        self.device=device
        self.dtype=dtype
        self.stream=stream_for(device)
        self.flat=torch.empty(sum(sizes),device=device,dtype=dtype)
        self.key=storage_key(self.flat)
        self.views=[v.view(s) for v,s in zip(self.flat.split(self.sizes),self.shapes,strict=True)]
        self.pending=False
        self.counts={"calls":0,"hits":0,"fallbacks":0,"cat_calls":0,"merged_boundaries":0,
                     "max_cat_inputs":0,"min_cat_inputs":None,"reused_view_steps":0}

    def pack(self,gradients):
        self.counts["calls"]+=1
        if self.pending:
            raise RuntimeError("previous ZeRO partition has not reached the optimizer")
        if (stream_for(self.device)!=self.stream or len(gradients)!=len(self.shapes) or
                any(type(g) is not torch.Tensor or tuple(g.shape)!=s or g.dtype!=self.dtype or g.device!=self.device
                    or not g.is_contiguous() or g.layout!=torch.strided or g.is_neg() or g.is_conj() or g.requires_grad
                    for g,s in zip(gradients,self.shapes))):
            self.counts["fallbacks"]+=1
            return None
        pieces,merged=segments(gradients,self.sizes,self.key)
        if not pieces:
            self.counts["fallbacks"]+=1
            return None
        return self.commit_pieces(pieces,merged)

    def commit_pieces(self,pieces,merged):
        with torch.autograd.profiler.record_function("EagerPartitionPackCat"):
            torch.cat(pieces,out=self.flat)
        self.pending=True
        self.counts["hits"]+=1
        self.counts["cat_calls"]+=1
        self.counts["merged_boundaries"]+=merged
        self.counts["max_cat_inputs"]=max(self.counts["max_cat_inputs"],len(pieces))
        minimum=self.counts["min_cat_inputs"]
        self.counts["min_cat_inputs"]=len(pieces) if minimum is None else min(minimum,len(pieces))
        self.counts["reused_view_steps"]+=int(self.counts["hits"]>1)
        return self.views,self.flat

    def before_step(self):
        if self.pending and stream_for(self.device)!=self.stream:
            raise RuntimeError("cannot reuse a private flat partition across optimizer streams")
        self.pending=False
