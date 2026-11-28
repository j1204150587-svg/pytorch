"""Private forward adapter enabled only for the stock HSTU BF16 eager training layout."""
import json
import os
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def load():
    path = Path(os.environ["NPU_RECSDK_LIBRARY"]).resolve(strict=True)
    torch.ops.load_library(str(path))
    return str(path)


class Experiment:
    def __init__(self):
        self.mode=os.environ["NPU_RECSDK_EAGER_ORDER"]
        if self.mode not in ("0","1"):
            raise ValueError(self.mode)
        self.library=load()
        self.calls=self.hits=self.fallbacks=0

    def __call__(self,q,k,v,alpha):
        self.calls+=1
        if self.mode=="0":
            return None
        b,h,l,d=q.shape
        if not (alpha==.25 and 1<=b<=4 and 12<=l<=32 and (h,d)==(4,16) and torch.is_grad_enabled()
                and all(x.device==q.device and x.device.type=="npu" and x.dtype==torch.bfloat16
                        and x.shape==q.shape and x.stride()[1:]==(16,256,1) for x in (q,k,v))):
            self.fallbacks+=1
            return None
        self.hits+=1
        with torch.profiler.record_function("HSTURecSDKDenseAttention"):
            return torch.ops.mxrec.hstu_dense(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),None,None,0,l,0.)


def receipt(blocks,rank_dir,steps):
    rows=[vars(block.recsdk) for block in blocks]
    valid=all(row["calls"]==steps and row["hits"]==(steps if row["mode"]=="1" else 0) and not row["fallbacks"] for row in rows)
    obj=dict(blocks=rows,mode=rows[0]["mode"],num_blocks=len(rows),passed=valid)
    (rank_dir/"recsdk_exact.json").write_text(json.dumps(obj,indent=2)+"\n")
