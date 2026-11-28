"""Compare all single-block gradients, including the input's independent residual edges."""
import copy
import importlib.util
import sys
from types import MethodType

import torch


def full_block(model_file, operation, compare, device):
    spec=importlib.util.spec_from_file_location("hstu_original_block_for_native_rms",str(model_file))
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    results=[]
    def fast(norm,x):
        return operation(x,norm.weight,norm.eps)
    for b,l in ((1,19),(2,14),(2,22),(4,31)):
        torch.manual_seed(2000+b+l)
        baseline=module.HSTUBlock(64,4,16,16,0.0).to(dtype=torch.bfloat16,device=device)
        changed=copy.deepcopy(baseline)
        changed.input_norm.forward=MethodType(fast,changed.input_norm)
        changed.output_norm.forward=MethodType(fast,changed.output_norm)
        x=torch.randn((b,l,64)).to(dtype=torch.bfloat16,device=device).requires_grad_()
        z=x.detach().clone().requires_grad_()
        a,c=baseline(x),changed(z)
        up=torch.randn(a.shape).to(device)
        ga,gb=[torch.autograd.grad((o.float()*up).sum(),(i,*mod.parameters()))
               for mod,i,o in ((baseline,x,a),(changed,z,c))]
        checks={name:compare(v,w) for name,v,w in zip(("y","dx",*dict(baseline.named_parameters())),(a,*ga),(c,*gb))}
        results.append({"shape":[b,l],"components":checks,"equal":all(not v["different"] for v in checks.values())})
    return results
