"""Check the complete four-block HSTU model and 1,024 independent table gradients."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import MethodType

import torch
from torch.nn import functional as F


def run(model_file,library):
    torch.set_num_threads(2)
    torch.manual_seed(1234)
    torch.ops.load_library(str(library.resolve()))
    spec=importlib.util.spec_from_file_location("original_hstu_full_model",str(model_file))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    cfg=mod.ModelConfig(num_tables=1024,table_rows=64)
    a=mod.MultiTableHSTU(cfg).bfloat16()
    b=copy.deepcopy(a)
    def fast(norm,x):
        return torch.ops.npu_hstu_native_rms.rms.default(x,norm.weight,norm.eps)
    for m in b.modules():
        if isinstance(m,mod.RMSNorm):
            m.forward=MethodType(fast,m)
    def bits(t):
        return t.detach().contiguous().view(torch.int16)
    names=list(dict(a.named_parameters()))
    rows=[]
    for batch,length in ((1,19),(2,14),(2,22),(4,31)):
        ids=torch.randint(1,cfg.vocab_size,(batch,length+1))
        items,targets=ids[:,:-1].contiguous(),ids[:,1:].contiguous()
        groups=torch.randint(0,cfg.num_user_groups,(batch,))
        feat=torch.randint(0,cfg.table_rows,(batch,length,cfg.num_tables,cfg.features_per_table))
        def fwd_bwd(m):
            logits=m(items,groups,feat)
            loss=F.cross_entropy(logits.float().reshape(-1,cfg.vocab_size),targets.reshape(-1))
            grads=torch.autograd.grad(loss,tuple(m.parameters()))
            return logits.detach(),loss.detach(),grads
        ya,la,ga=fwd_bwd(a)
        yb,lb,gb=fwd_bwd(b)
        differences=[]
        for name,x,y in zip(names,ga,gb):
            neq=bits(x)!=bits(y)
            if neq.any():
                differences.append({"name":name,"different":int(neq.sum()),
                                    "max_abs":float((x.float()-y.float()).abs().max())})
        rows.append({"shape":[batch,length],"logits_bit_differences":int((bits(ya)!=bits(yb)).sum()),
                     "loss":float(la),"loss_equal":torch.equal(la.view(torch.int32),lb.view(torch.int32)),
                     "different_parameter_gradients":differences})
    return {"device":"cpu","torch":str(torch.__version__),"parameters":len(names),
            "gradient_elements":sum(p.numel() for p in a.parameters()),"shapes":rows,
            "pass":all(not s["logits_bit_differences"] and s["loss_equal"]
                       and not s["different_parameter_gradients"] for s in rows)}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-file",type=Path,required=True)
    p.add_argument("--library",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    report=run(args.model_file,args.library)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(report["pass"],report["parameters"],report["gradient_elements"])
