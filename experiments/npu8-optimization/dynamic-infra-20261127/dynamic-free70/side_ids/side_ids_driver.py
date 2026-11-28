"""ABI witness shared with the verified eager fold (same driver source/cache checks)."""
import hashlib
from pathlib import Path
import re

from triton.compiler.compiler import CompiledKernel

# Select the target runtime's ABI once; hook registrations change at runtime.
try:
    from triton import knobs
    HOOK_OWNER = knobs.runtime
except ImportError:
    HOOK_OWNER = CompiledKernel


def values(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    return vars(obj)


def active_hooks(binary):
    for name in ("launch_enter_hook", "launch_exit_hook"):
        h = getattr(HOOK_OWNER, name, None)
        if h is not None and bool(getattr(h, "calls", (h,))):
            return True
    return False


def witness_eager_launcher(binary):
    """Regenerate the actual NPULauncher C++ and require its loaded content-addressed .so."""
    from triton.backends.ascend import driver
    original = binary.run
    actual = Path(original.get_launcher_so_path()).resolve()
    key = lambda arg: binary.src.fn.arg_names.index(arg) if isinstance(arg, str) else arg
    constants = {key(k): v for k,v in binary.src.constants.items()}
    signature = {key(k): v for k,v in binary.src.signature.items()}
    body = driver.generate_npu_wrapper_src(constants, signature, binary.metadata)
    header = driver.generate_npu_header_src()
    source_hash = hashlib.sha256((header + "\0" + body).encode()).hexdigest()
    found = driver.get_cache_manager(source_hash).get_file(actual.name)
    if not found or Path(found).resolve() != actual:
        raise ValueError("regenerated driver C++ cache key does not match the loaded launcher")
    caps = set(re.findall(r"blockNum\s*=\s*std::min\(blockNum,\s*\(uint32_t\)(\d+)\)", body))
    if len(caps) > 1 or ("std::min(blockNum" in body and not caps):
        raise ValueError("unresolved physical block cap")
    if "rtKernelLaunch(func, blockNum," not in body:
        raise ValueError("unrecognized baseline launch API")
    return {"original_shared_object":str(actual), "shared_object_sha256":hashlib.sha256(actual.read_bytes()).hexdigest(),
            "driver_py_sha256":hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest(),
            "launcher_source_cache_hash":source_hash,
            "allocates_workspace":bool(re.search(r"uint64_t\s+totalWorkSpaceSize\s*=", body)),
            "allocates_sync_lock":bool(re.search(r"uint64_t\s+syncBlockLockSize\s*=", body)),
            "actual_null_resource_slots":bool(re.search(r"nullptr,\s*nullptr,\s*static_cast<void\*>\(arg0\)", body)),
            "uses_ffts":"ret = rtGetC2cCtrlAddr(" in body, "auto_map_physical_cap":int(next(iter(caps))) if caps else None}
