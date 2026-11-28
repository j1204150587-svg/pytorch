"""Exercise the compiled query and mutable PTA registry without opening a device."""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

here = Path(__file__).resolve().parent
receipt = json.loads((here/"build/build-receipt.json").read_bytes())
if len(sys.argv) > 1:
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    import torch
    assert all(not Path(f"/dev/davinci{i}").exists() for i in range(16))
    target = (here/"cpu-lib/libtorch_npu.so").resolve()
    module = C.CDLL(str(target), mode=C.RTLD_GLOBAL)
    mapped = {Path(s.split()[-1]).resolve() for s in Path("/proc/self/maps").read_text().splitlines()
              if s.endswith("/libtorch_npu.so")}
    assert mapped == {target}
    assert hashlib.sha256(target.read_bytes()).hexdigest() == json.loads((here/"cpu-receipt.json").read_bytes())["gate_sha256"]
    check = module.option_test_value
    check.argtypes, check.restype = [C.c_char_p], C.c_long
    timer = module.option_test_time
    timer.argtypes, timer.restype = [C.c_uint,C.POINTER(C.c_uint)], C.c_uint64
    initial = os.environ.get("TORCH_NPU_USE_COMPATIBLE_IMPL")
    assert check(None) == 4 * (initial == "1")
    for value in (b"1", b"0", b"", b"11", b"1", b"off", b"1"):
        assert check(value) == 4 * (value == b"1")
    n = 100000
    truth = C.c_uint()
    windows = [timer(n, C.byref(truth))/n for _ in range(16)]
    assert truth.value == n
    print(json.dumps(dict(key=os.environ["TORCH_NPU_STABLE_COMPATIBLE_KEY"], initial=initial,
               trials_ns=windows[1:],median_ns=statistics.median(windows[1:]),n=n,cpu=list(os.sched_getaffinity(0)))))
    sys.exit(0)
unit = receipt["compile"]
cmd = list(unit["argv"])
obj = here/"gate.o"
cmd[cmd.index("-c")+1] = str(here/"gate.cc")
cmd[cmd.index("-o")+1] = str(obj)
if "-MF" in cmd:
    cmd[cmd.index("-MF")+1] = str(here/"gate.d")
subprocess.run(cmd, cwd=unit["cwd"], check=True)
link = list(receipt["link"])
target = here/"cpu-lib/libtorch_npu.so"
target.parent.mkdir(exist_ok=True)
link[link.index("-o")+1] = str(target)
link.insert(link.index(unit["object"])+1,str(obj))
checkout = Path(json.loads((here.parents[1]/"embedding/build/build-receipt.json").read_bytes())["link_cwd"])
subprocess.run(link,cwd=checkout,check=True)
(here/"cpu-receipt.json").write_text(json.dumps(dict(training_sha256=receipt["sha256"],
    added_test_object_sha256=hashlib.sha256(obj.read_bytes()).hexdigest(),
    gate_sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2))
cpu = min(os.sched_getaffinity(0))
os.sched_setaffinity(0,{cpu})
results = []
for key,initial in (("0",None),("1",None),("1","1"),("0","1")):
    env = dict(os.environ,TORCH_NPU_STABLE_COMPATIBLE_KEY=key)
    env.pop("TORCH_NPU_USE_COMPATIBLE_IMPL",None)
    if initial is not None:
        env["TORCH_NPU_USE_COMPATIBLE_IMPL"] = initial
    raw = subprocess.check_output([sys.executable,"-B",__file__,"child"],env=env,text=True)
    results.append(json.loads(raw.splitlines()[-1]))
(here/"cpu.json").write_text(json.dumps(results,indent=2))
print([(r["key"],r["median_ns"]) for r in results])
