"""Record the actual matmul input/output geometry during an eager CPU HSTU backward."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import torch
from torch.utils._python_dispatch import TorchDispatchMode


def geometry(t,root,compact):
    n=t.untyped_storage().nbytes()//t.element_size()
    off=t.storage_offset()
    if compact and t.untyped_storage()._cdata==root and t.ndim==2 and t.dtype==torch.bfloat16 and (
        tuple(t.shape) in ((256,64),(64,192)) and t.is_contiguous() and off*2%128==0):
        off,n=0,t.numel()
    return (str(t.dtype),tuple(t.shape),t.stride(),off,n)


def main(args):
    sys.path.insert(0,str(args.workspace))
    from model import HSTUBlock
    layout=json.loads(args.layout.read_bytes())
    shapes=json.loads(args.schedule.read_bytes())
    torch.manual_seed(1234)
    torch.set_num_threads(2)
    blocks=torch.nn.ModuleList(HSTUBlock(64,4,16,16,0.0).bfloat16() for _ in range(4))
    flat=torch.randn(layout["flat"]["elements"],dtype=torch.bfloat16)
    for r in layout["projection_views"]:
        layer=getattr(blocks[r["layer"]],r["projection"])
        g,b=r["transposed_weight"],r["bias"]
        layer.weight.data=flat.as_strided(g["shape"],g["stride"],g["storage_element_offset"]).t()
        layer.bias.data=flat.as_strided(b["shape"],b["stride"],b["storage_element_offset"])
    root=flat.untyped_storage()._cdata
    data=[]
    class Capture(TorchDispatchMode):
        def __torch_dispatch__(self,func,types,args=(),kwargs=None):
            out=func(*args,**(kwargs or {}))
            if func is torch.ops.aten.mm.default:
                ts=[*args,out]
                has_weight=any(t.untyped_storage()._cdata==root for t in args)
                data.append(dict(step=step,shape=[b,l],has_projection_parameter=has_weight,
                                 original=[geometry(t,root,False) for t in ts],
                                 canonical=[geometry(t,root,True) for t in ts]))
            return out
    for step,(b,l) in enumerate(shapes):
        blocks.zero_grad(set_to_none=True)
        x=torch.randn(b,l,64,dtype=torch.bfloat16,requires_grad=True)
        for block in blocks:
            x=block(x)
        upstream=torch.randn_like(x)
        with Capture():
            x.backward(upstream)
    result={}
    for name,rows in (("all",data),("read_only_weight",[r for r in data if r["has_projection_parameter"]])):
        result[name]={}
        for mode in ("original","canonical"):
            keys=[json.dumps(r[mode],separators=(",",":")) for r in rows]
            unique=len(set(keys))
            first=set()
            hits=0
            for key,row in zip(keys,rows):
                if key in first and row["step"]>=10:
                    hits+=1
                first.add(key)
            result[name][mode]=dict(calls=len(rows),unique=unique,repeats_in_measured_50=hits)
    args.output.write_text(json.dumps(dict(summary=result,per_call=data),separators=(",",":")))
    print(json.dumps(result))


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for f in ("layout","schedule","workspace","output"):
        cli.add_argument("--"+f,required=True,type=Path)
    main(cli.parse_args())
