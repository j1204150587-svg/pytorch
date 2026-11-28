import ctypes
import json
import os
from pathlib import Path


def load():
    paths={s.split()[-1] for s in Path("/proc/self/maps").read_text().splitlines() if s.rstrip().endswith("/libtorch_npu.so")}
    if len(paths)!=1:raise ValueError(f"ambiguous loaded libtorch_npu: {paths}")
    path=paths.pop()
    counter=ctypes.CDLL(path).hstu_hccl_rank1_count
    counter.argtypes=[ctypes.c_uint]
    counter.restype=ctypes.c_uint64
    return path,counter


class Counts:
    def __init__(self):
        self.path,self.counter=load()
        self.start=[self.counter(i) for i in (0,1)]

    def finish(self,rank,steps):
        hits,health=[self.counter(i)-v for i,v in enumerate(self.start)]
        mode=os.environ["NPU_HCCL_RANK1_SCALAR"]
        out=dict(mode=mode,steps=steps,hits=hits,health_fallback=health,library=self.path,
                 passed=hits==(steps if mode=="intmax" else 0) and health==0)
        (rank/"hccl_rank1.json").write_text(json.dumps(out,indent=2)+"\n")
