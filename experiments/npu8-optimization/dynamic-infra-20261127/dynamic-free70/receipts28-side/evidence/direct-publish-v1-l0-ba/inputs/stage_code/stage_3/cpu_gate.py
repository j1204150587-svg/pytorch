"""Check ordered segments, shared aliases, BFloat16 bytes and fallback without an NPU."""
from pathlib import Path
import json
import sys

import torch
from partition_pack import GradientPack,segments,storage_key


def target(slices,device):
    return GradientPack([g.shape for g in slices],[g.numel() for g in slices],device,torch.bfloat16)


def main():
    torch.set_num_threads(1)
    device=torch.device("cpu")
    slab=torch.arange(1052*8+19,dtype=torch.int32).remainder(31).to(torch.bfloat16)
    side=list(slab.narrow(0,9,1024*8).view(1024,2,4).unbind())
    head=torch.tensor([2,4,7],dtype=torch.bfloat16)
    tail=torch.tensor([1],dtype=torch.bfloat16)
    grads=[head,*side,tail]
    pack=target(grads,device)
    def check_values(gs):
        views,flat=pack.pack(gs)
        expected=torch.cat([g.reshape(-1) for g in gs])
        assert torch.equal(flat,expected) and all(torch.equal(a,b) for a,b in zip(views,gs,strict=True))
        assert all(storage_key(v)==storage_key(flat) for v in views)
        return views
    first=check_values(grads)
    assert pack.counts["merged_boundaries"]==1023 and pack.counts["max_cat_inputs"]==3
    first[-1].fill_(float("inf"))
    assert torch.isinf(pack.flat[-1])
    try:
        pack.pack(grads)
    except RuntimeError:
        pass
    else:
        raise AssertionError("double publication")
    pack.before_step()
    side[0].add_(2)
    second=check_values(grads)
    assert first is second and all(a is b for a,b in zip(first,second))
    pack.before_step()
    facets=[slab[4:8],slab[12:16],slab[8:12],slab[8:12],slab[10:14],slab.clone()[14:18]]
    pieces,shares=segments(facets,[4]*len(facets),None)
    assert shares==0 and len(pieces)==6 and torch.equal(torch.cat(pieces),torch.cat(facets))
    assert pack.pack([*(g for g in grads[:-1]),pack.views[-1]]) is None
    assert pack.pack([*(g for g in grads[:-1]),torch.ones(2,dtype=torch.bfloat16)]) is None
    assert pack.pack([*(g for g in grads[:-1]),torch.ones(1,dtype=torch.float32)]) is None
    assert pack.counts["cat_calls"]==2
    result={"ordered_1024_shared_inputs":True,"output_identity_reused":True,"reordered_overlapped_separate":True,
            "overlap_wrong_shape_wrong_dtype_fallbacks":3,"pack_stats":pack.counts}
    output=json.dumps(result,indent=2)
    if len(sys.argv)>1:
        Path(sys.argv[1]).write_text(output+"\n",encoding="utf-8")
    print(output)


if __name__=="__main__":
    main()
