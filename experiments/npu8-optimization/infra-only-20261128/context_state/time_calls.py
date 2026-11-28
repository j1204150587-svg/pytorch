"""Measure context queries in C on real initialized device 0, with counters and profiler disabled."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import subprocess
from threading import Barrier


def main():
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    p.add_argument("--switch",required=True,choices=("0","1"))
    cfg=p.parse_args()
    os.environ.update(NPU_CONTEXT_POSITIVE_MASK=cfg.switch,NPU_DYNAMIC_REBASE_COUNT="0")
    import torch
    import torch_npu
    torch.npu.set_device(0)
    assert torch.ones(16,device="npu").sum().item()==16
    torch.npu.synchronize()
    path,={r.split()[-1] for r in Path("/proc/self/maps").read_text().splitlines() if r.split()[-1].endswith("/libtorch_npu.so")}
    lib=ctypes.CDLL(path)
    syms=subprocess.check_output(["nm","-C",path],text=True).splitlines()
    def off(name):
        row,=[s for s in syms if s.endswith(" "+name)]
        return int(row.split()[0],16)
    base=ctypes.cast(lib.npu_active_context_counter,ctypes.c_void_p).value-off("npu_active_context_counter")
    ptr=base+off("c10_npu::isDeviceCtxActive(int)")
    fn=ctypes.CDLL(str(Path(__file__).with_name("time_calls.so"))).time_context_calls
    fn.argtypes=(ctypes.c_void_p,ctypes.c_uint,ctypes.c_uint,ctypes.POINTER(ctypes.c_uint64))
    fn.restype=ctypes.c_int
    result={}
    for n in (1,2,4):
        barrier=Barrier(n)
        def run(i):
            values=(ctypes.c_uint64*32)()
            assert fn(ptr,4,1024,values)==0
            barrier.wait()
            assert fn(ptr,32,2048,values)==0
            return list(values)
        with ThreadPoolExecutor(max_workers=n) as pool:result[n]=list(pool.map(run,range(n)))
    cfg.output.write_text(json.dumps(dict(switch=cfg.switch,library=path,sample_size=2048,
        per_thread_group_duration_ns=result),indent=2)+"\n")


if __name__=="__main__":
    main()
