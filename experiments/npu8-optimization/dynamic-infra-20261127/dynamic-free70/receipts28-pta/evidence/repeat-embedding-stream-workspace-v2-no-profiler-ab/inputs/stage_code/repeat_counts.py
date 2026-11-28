"""Verify that all 1,024 side-table calls per training step use the intended route."""
import ctypes
import json
import os
from pathlib import Path


COUNTERS = ("eligible", "fallback", "prepared", "reused", "launches", "drained", "repeat_rejected",
            "rebind_failure", "live", "peak_slots", "peak_workspace_bytes", "runtime_fallback")


def write_repeat_counts(rank, steps):
    paths = {s.split()[-1] for s in Path("/proc/self/maps").read_text().splitlines() if "/libtorch_npu.so" in s}
    if len(paths) != 1:
        raise ValueError(f"unexpected loaded TorchNPU libraries {paths}")
    library = paths.pop()
    count = ctypes.CDLL(library).npu_dynamic_repeat_counter
    count.argtypes = [ctypes.c_uint]
    count.restype = ctypes.c_uint64
    seen = {name:int(count(i)) for i,name in enumerate(COUNTERS)}
    mode = os.environ["NPU_DYNAMIC_REPEAT_EMBEDDING"]
    expected = dict(eligible=1024*steps, fallback=3*steps, launches=1024*steps,
                    prepared=(16 if mode=="repeat" else 1024)*steps,
                    reused=(1008 if mode=="repeat" else 0)*steps, runtime_fallback=0,
                    rebind_failure=0, repeat_rejected=0, peak_slots=16 if mode=="repeat" else 0)
    if mode == "0":
        expected = dict.fromkeys(COUNTERS,0)
    elif mode != "fresh" and mode != "repeat":
        raise ValueError(f"unexpected mode {mode}")
    passed = all(seen[k] == v for k,v in expected.items()) and (mode=="0" or seen["peak_workspace_bytes"]>0)
    data = dict(library=library, mode=mode, expected=expected, actual=seen, steps=steps, passed=passed)
    (rank / "repeat_embedding_counts.json").write_text(json.dumps(data, indent=2)+"\n")
