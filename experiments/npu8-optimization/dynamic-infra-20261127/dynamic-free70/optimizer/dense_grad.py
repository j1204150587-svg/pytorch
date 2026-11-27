"""Experimental eager side-table weight-gradient kernel. No graph replay or temporary full lookup."""
import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def grouped_dense_backward(ids, upstream):
    """Read [B,L,T,4] local int64 IDs and [B,L,64] BF16 d(side), return [T,64,64]."""
    if (ids.ndim != 4 or ids.shape[-1] != 4 or ids.dtype != torch.int64 or
            upstream.shape != (*ids.shape[:2],64) or upstream.dtype != torch.bfloat16 or
            ids.device.type != "npu" or ids.device != upstream.device or not ids.is_contiguous() or
            upstream.requires_grad or not (1 <= ids.shape[0] <= 4 and 1 <= ids.shape[1] <= 32)):
        raise ValueError("unsupported side-gradient inputs")
    # Mean over exactly four features divides the upstream by four in BF16.
    # The table-local row IDs must already have passed the paired forward lookup.
    from dense_grad_kernel import side_grad
    batch,length,tables,_ = ids.shape
    grad = upstream.contiguous()
    out = torch.empty((tables,64,64),dtype=grad.dtype,device=grad.device)
    side_grad[(tables,)](ids,grad,out,batch*length,tables,16)
    return out


class GroupedSideGrad(torch.autograd.Function):
    """Feed an existing flat side-weight into the ordinary grouped forward."""
    @staticmethod
    def forward(ctx,weight,ids,offsets):
        from side_tables_op import OrderedTableSum
        tables=ids.shape[2]
        if (tuple(weight.shape)!=(tables*64,64) or weight.dtype!=torch.bfloat16 or
                ids.ndim!=4 or ids.shape[3]!=4 or ids.device!=weight.device or
                tuple(offsets.shape)!=(tables,1,1,1) or offsets.device!=ids.device):
            raise ValueError("unsupported side table bank")
        local=ids.movedim(2,0)
        rows=torch.where((local>=0)&(local<64),local+offsets,tables*64).contiguous()
        means=F.embedding(rows,weight).mean(dim=-2)
        ctx.save_for_backward(ids)
        return OrderedTableSum.apply(means)

    @staticmethod
    @once_differentiable
    def backward(ctx,grad):
        ids,=ctx.saved_tensors
        bank_grad=grouped_dense_backward(ids,grad)
        return bank_grad.view(-1,64),None,None
