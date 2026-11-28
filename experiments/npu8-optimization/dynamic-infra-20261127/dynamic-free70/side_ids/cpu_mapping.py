"""Independent CPU witness for the kernel's address mapping, offset use and integer conditions."""
import argparse
import json
from pathlib import Path

import torch
from side_ids_impl import reference


def evaluate(schedule):
    torch.set_num_threads(2)
    gen=torch.Generator().manual_seed(1234)
    rows=[]
    for dt in (torch.int32,torch.int64):
        for b,l in schedule:
            ids=torch.randint(-3,69,(b,l,1024,4),dtype=dt,generator=gen)
            lim=torch.iinfo(dt);ids.reshape(-1)[:8]=torch.tensor([lim.min,lim.max,0,1,1,63,64,-1],dtype=dt)
            off=(torch.randperm(1024,generator=gen,dtype=torch.int64)*64).view(1024,1,1,1)
            off.flatten()[:4]=torch.tensor([2**63-11,-2**63+1,-177,93],dtype=torch.int64)
            expected=reference(ids,off,64)
            # Run all logical (token, group=16, lane=256) positions in a single CPU tensor.
            lane=torch.arange(256)
            group=torch.arange(16).view(1,16,1)
            token=torch.arange(b*l).view(-1,1,1)
            table=group*64+lane//4
            inp=token*4096+table*4+(lane%4)
            dest=(table*(b*l)+token)*4+(lane%4)
            v=ids.flatten()[inp].long()
            valid=(v>=0)&(v<64)
            output=torch.empty(expected.numel(),dtype=torch.int64)
            output[dest.flatten()]=torch.where(valid,torch.where(valid,v,0)+off.flatten()[table],65536).flatten()
            mismatch=int((output.view_as(expected)!=expected).sum())
            rows.append({"shape":[b,l],"dtype":str(dt),"values":expected.numel(),"different":mismatch})
    return {"implementation":"independent CPU indexing model of the Triton grid; ATen reference","cases":rows,
            "all_passed":all(not r["different"] for r in rows)}


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__);cli.add_argument("schedule",type=Path);cli.add_argument("out",type=Path)
    a=cli.parse_args();result=evaluate(json.loads(a.schedule.read_bytes()))
    a.out.write_text(json.dumps(result,indent=2)+"\n");print(len(result["cases"]),result["all_passed"])
    raise SystemExit(0 if result["all_passed"] else 1)
