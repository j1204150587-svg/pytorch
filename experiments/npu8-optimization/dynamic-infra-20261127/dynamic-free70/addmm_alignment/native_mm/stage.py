"""Record accepted, original-path and descriptor construction counts in each training process."""
from pathlib import Path
import sys


if __name__ == "__main__":
    path = Path(sys.argv[1]) / "train_deepspeed.py"
    source = path.read_text()
    mark = "        end = time.perf_counter()\n"
    if source.count(mark) != 1:
        raise ValueError("training output marker changed")
    code = '''        import ctypes
        mm_library = Path(torch_npu.__file__).parent / "lib/libtorch_npu.so"
        mm_count = ctypes.CDLL(str(mm_library)).npu_mm_rebase_counter
        mm_count.argtypes = [ctypes.c_uint]
        mm_count.restype = ctypes.c_uint64
        accepted, original, keys, descriptors = [mm_count(i) for i in range(4)]
        mm_enabled = os.getenv("NPU_DYNAMIC_MM_REBASE_128") == "1"
        expected = 8 * args.steps if mm_enabled else 0
        mm_report = dict(enabled=mm_enabled, steps=args.steps, accepted=accepted, original=original,
                         keys=keys, descriptors=descriptors,
                         passed=accepted==keys==expected and (original>=expected if mm_enabled else original==0))
        (rank_dir / "mm_rebase.json").write_text(json.dumps(mm_report,indent=2))
'''
    new = source.replace(mark,mark+code)
    compile(new,str(path),"exec")
    path.write_text(new)
