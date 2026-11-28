"""Check every installed payload file, the actual no-device import and native ELF mapping before publication."""
import hashlib
import json
import os
from pathlib import Path
import sys

here=Path(__file__).resolve().parent
overlay=here/"overlay"
head="e3cda6399cb194c813530222478ea018f5aca5c9"
assert not Path("/dev/davinci_manager").exists() and not Path("/dev/davinci0").exists()
def sha(path):
    with path.open("rb") as fp:
        return hashlib.file_digest(fp,"sha256").hexdigest()
payload=json.loads((here/"package-payload.json").read_bytes())
installed=here/"wheel-install"
for name,data in payload.items():
    assert sha(installed/name)==data["package_sha256"],name
branch_files={name for name in payload if name.startswith("torch_npu/")}
actual_files={f.relative_to(overlay).as_posix() for f in (overlay/"torch_npu").rglob("*") if f.is_file()}
assert {f.name for f in overlay.iterdir()}=={"torch_npu"} and actual_files==branch_files,(actual_files-branch_files,branch_files-actual_files)
for name in branch_files:
    assert sha(overlay/name)==payload[name]["package_sha256"],name
live=Path("/training-stock")
live_files={"torch_npu/"+f.relative_to(live).as_posix() for f in live.rglob("*") if f.is_file() and "__pycache__" not in f.relative_to(live).parts}
assert live_files==branch_files,(live_files-branch_files,branch_files-live_files)
for name in branch_files:
    assert sha(live/Path(name).relative_to("torch_npu"))==payload[name]["stock_sha256"],name
stock_version=(live/"version.py").read_bytes()
branch_version=(overlay/"torch_npu/version.py").read_bytes()
base=b"38e483ee79ec8a3286517808740a811c0f0ae66c"
assert stock_version.count(base)==1 and stock_version.replace(base,head.encode())==branch_version
import torch
import torch_npu
import torch_npu.version
import ctypes
native=overlay/"torch_npu/lib/libtorch_npu.so"
cmod=Path(torch_npu._C.__file__).resolve()
assert Path(torch_npu.__file__).resolve().is_relative_to(overlay) and cmod.is_relative_to(overlay)
assert torch_npu.version.git_version==head
libs=sorted({row.split()[-1] for row in Path("/proc/self/maps").read_text().splitlines() if row.split()[-1].endswith("/libtorch_npu.so")})
assert libs==[str(native)],libs
dll=ctypes.CDLL(str(native))
symbols=['npu_embedding_forward_rebase_counter','npu_embedding_backward_rebase_counter']
from mmap import mmap, ACCESS_READ
with native.open("rb") as fp, mmap(fp.fileno(),0,access=ACCESS_READ) as data:
    for marker in (b"TORCH_NPU_QUEUE_WAKE_AUDIT",b"TORCH_NPU_QUEUE_NOTIFY_FULL_ONLY",b"PTA_QUEUE_PRODUCER_WAKE device="):
        assert data.find(marker)>=0,marker
for sym in symbols:
    getattr(dll,sym)
r=json.loads((here/"wheel-receipt.json").read_bytes())
result=dict(commit=head,sha256=sha(native),inner_package=str(overlay),inner_elf=str(native),wheel=r["wheel"],wheel_sha256=r["wheel_sha256"],
    version_sha256=sha(overlay/"torch_npu/version.py"),stock_version_sha256=sha(live/"version.py"),stock_elf_sha256=r["stock_elf_sha256"],
    wheel_install=str(installed),torch_npu_wheel_files_verified=len(branch_files),overlay_matches_all_wheel_package_files=True,
    training_stock_changes=["torch_npu/lib/libtorch_npu.so","torch_npu/version.py"],overlay_pyc_files=0,
    host_root="/data/recsys-gr-ds019-host88/artifacts/infra-only-20261128/branch-stability-v210/build",validated_payload=len(payload),
    torch_version=torch.__version__,torch_npu_version=torch_npu.__version__,torch_npu_python=str(torch_npu.__file__),
    torch_npu_c_extension=str(cmod),mapped_libraries=libs,exports=symbols,hardware_smoke="CPU-only import and dlopen; no NPU device visible")
assert result["sha256"]==r["library_sha256"] and sha(Path(r["wheel"]))==r["wheel_sha256"]
tmp=here/"ready.json.new"
tmp.write_text(json.dumps(result,indent=2)+"\n")
os.replace(tmp,here/"ready.json")
print(json.dumps(result,indent=2))
