"""Snapshot counters after the existing synchronization at each end of the run."""
from pathlib import Path
import sys


def stage(root):
    path = root / "train_deepspeed.py"
    text = path.read_text()
    before = "    start = time.perf_counter()\n"
    after = "        end = time.perf_counter()\n"
    for mark in (before, after):
        if text.count(mark) != 1:
            raise ValueError("unrecognized synchronized window")
    text = text.replace(before, '''    import ctypes
    import resource
    queue_library = Path(torch_npu.__file__).resolve().parent / "lib/libtorch_npu.so"
    queue_counter = ctypes.CDLL(str(queue_library)).npu_hstu_short_poll_counter
    queue_counter.argtypes = [ctypes.c_uint]
    queue_counter.restype = ctypes.c_uint64
    queue_fields = ("empty", "poll", "hit", "sleep", "notify")
    queue_before = [queue_counter(i) for i in range(len(queue_fields))]
    cpu_before = resource.getrusage(resource.RUSAGE_SELF)
''' + before)
    text = text.replace(after, after + '''        cpu_after = resource.getrusage(resource.RUSAGE_SELF)
        queue_delta = {key: queue_counter(i) - queue_before[i] for i,key in enumerate(queue_fields)}
        queue_delta.update(triggered=queue_delta["hit"] > 0, slept=queue_delta["sleep"] > 0,
                           enabled=os.environ.get("NPU_DYNAMIC_SHORT_POLL") == "1",
                           library=str(queue_library),
                           process_cpu_s=cpu_after.ru_utime + cpu_after.ru_stime -
                                         cpu_before.ru_utime - cpu_before.ru_stime)
        (rank_dir / "queue_short_poll.json").write_text(json.dumps(queue_delta, indent=2))
''')
    compile(text, str(path), "exec")
    path.write_text(text)


if __name__ == "__main__":
    stage(Path(sys.argv[1]))
