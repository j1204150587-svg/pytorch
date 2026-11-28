"""Capture counters for the two read-only descriptors at the end of training."""
from pathlib import Path
import sys


def stage(workspace):
    path=workspace/"train_deepspeed.py"
    source=path.read_text()
    anchor="        end = time.perf_counter()\n"
    if source.count(anchor)!=1:
        raise ValueError("unexpected training loop boundary")
    code='''        import ctypes
        addmm_library = Path(torch_npu.__file__).parent / "lib/libtorch_npu.so"
        addmm_count = ctypes.CDLL(str(addmm_library)).npu_addmm_rebase_counter
        addmm_count.argtypes = [ctypes.c_uint]
        addmm_count.restype = ctypes.c_uint64
        eligible, rejected, key_calls, desc_calls = [addmm_count(i) for i in range(4)]
        enabled = os.environ.get("NPU_DYNAMIC_ADDMM_REBASE_128") == "1"
        expected = 8 * len(metrics) if enabled else 0
        result = dict(eligible=eligible,rejected=rejected,key_calls=key_calls,desc_calls=desc_calls,
                      enabled=enabled,steps=len(metrics),
                      passed=eligible==expected and rejected==0 and key_calls==2*expected)
        (rank_dir/"addmm_rebase.json").write_text(json.dumps(result,indent=2))
'''
    new=source.replace(anchor,anchor+code)
    compile(new,str(path),"exec")
    path.write_text(new)


if __name__=="__main__":
    stage(Path(sys.argv[1]))
