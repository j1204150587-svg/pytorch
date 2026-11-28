"""Use only the assigned logical NPU; exercise normal and compatibility queries without resets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import json
import os
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser()
    p.add_argument("output",type=Path)
    p.add_argument("--switch",required=True,choices=("0","1"))
    p.add_argument("--compatible",required=True,choices=("0","1"))
    args=p.parse_args()
    os.environ.update(NPU_CONTEXT_POSITIVE_MASK=args.switch,NPU_DYNAMIC_REBASE_COUNT="1",
        TORCH_NPU_USE_COMPATIBLE_IMPL=args.compatible)
    import torch
    import torch_npu
    torch.npu.set_device(0)
    assert torch.ones(16,device="npu").sum().item()==16
    torch.npu.synchronize()
    path,={r.split()[-1] for r in Path("/proc/self/maps").read_text().splitlines() if r.split()[-1].endswith("/libtorch_npu.so")}
    lib=ctypes.CDLL(path)
    counter=lib.npu_active_context_counter
    counter.argtypes=(ctypes.c_uint,)
    counter.restype=ctypes.c_uint64
    # This native query has local ELF binding. The same image contains the exported counter,
    # so its runtime address provides the load bias without any assumption about /proc maps.
    symbols=subprocess.check_output(["nm","-C",path],text=True).splitlines()
    def offset(name):
        row,=[line for line in symbols if line.endswith(" "+name)]
        return int(row.split()[0],16)
    bias=ctypes.cast(counter,ctypes.c_void_p).value-offset("npu_active_context_counter")
    probe=ctypes.CFUNCTYPE(ctypes.c_bool,ctypes.c_int32)(bias+offset("c10_npu::isDeviceCtxActive(int)"))
    def snapshot():return [counter(i) for i in range(4)]
    assert probe(0)
    before=snapshot()
    def thread_read(_):return all(probe(0) for i in range(512))
    with ThreadPoolExecutor(max_workers=4) as ex:assert all(ex.map(thread_read, range(4)))
    delta=[b-a for a,b in zip(before,snapshot())]
    expect=[0,0,0,2048] if args.compatible=="1" else ([2048,0,0,0] if args.switch=="1" else [0,2048,0,0])
    assert delta==expect,(delta,expect)
    if args.compatible=="0":
        assert not any(probe(i) for i in (1,63,64,127,-1))
    args.output.write_text(json.dumps(dict(switch=args.switch,compatible=args.compatible,library=path,
        positive_reads=2048,threads=4,counts=delta,hardware_reset=False),indent=2)+"\n")


if __name__=="__main__":
    main()
