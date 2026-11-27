"""Guards for the opt-in whole-block C++ eager implementation."""
import torch
from torch import nn
from torch.nn.modules import module as module_impl


GLOBAL_HOOKS = tuple(getattr(module_impl,n,{}) for n in ("_global_forward_hooks","_global_forward_pre_hooks",
                     "_global_backward_hooks","_global_backward_pre_hooks"))


def eligible(block,x):
    if (block.num_heads!=4 or block.hidden_dim!=16 or block.attention_dim!=16 or block.embedding_dim!=64
            or block.attn_alpha!=0.25
            or x.dtype!=torch.bfloat16 or x.ndim!=3 or x.shape[-1]!=64 or not block.training
            or type(block.dropout) is not nn.Dropout or block.dropout.p!=0 or not torch.is_grad_enabled()
            or any(GLOBAL_HOOKS)):
        return False
    children=(block.input_norm,block.uvqk,block.output_norm,block.output,block.dropout)
    if any(m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks
           or "forward" in m.__dict__ for m in children):
        return False
    return (type(block.uvqk) is nn.Linear and type(block.output) is nn.Linear
            and block.uvqk.bias is not None and block.output.bias is not None)


def run(m,x):
    return torch.ops.npu_hstu_native_block.block.default(x,m.input_norm.weight,m.uvqk.weight,m.uvqk.bias,
              m.output_norm.weight,m.output.weight,m.output.bias,m.input_norm.eps,m.output_norm.eps)
