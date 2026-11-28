"""Read the installed host-side cache limit with a fresh process; no ACL/NPU initialization."""
import ctypes
import hashlib
import json
import os
from pathlib import Path

assert not Path("/dev/davinci_manager").exists() and not Path("/dev/davinci0").exists()
base=Path(os.environ["ASCEND_HOME_PATH"])/"aarch64-linux/lib64"
library=base/"libnnopbase.so"
nn=ctypes.CDLL(str(library))
mm=ctypes.CDLL(str(base/"libmmpa.so"))
mm.mmSysGetEnv.argtypes=[ctypes.c_int]
mm.mmSysGetEnv.restype=ctypes.c_char_p
platform=mm.mmSysGetEnv(2000)
read_limit=getattr(nn,"_ZN2op8internal18OpExecCacheManager17GetCacheSizeLimitEv")
read_limit.argtypes=[ctypes.c_void_p]
read_limit.restype=ctypes.c_size_t
get_manager=getattr(nn,"_ZN2op8internal21GetOpExecCacheManagerEv")
get_manager.argtypes=[]
get_manager.restype=ctypes.c_void_p
release=getattr(nn,"_ZN2op8internal25ReleaseOpExecCacheManagerEPv")
release.argtypes=[ctypes.c_void_p]
release.restype=None
ptr=get_manager()
assert ptr
try:
    effective=read_limit(ptr)
finally:
    release(ptr)
print(json.dumps(dict(environment=os.environ.get("ACLNN_CACHE_LIMIT"),
    mmSysGetEnv_2000=platform.decode() if platform is not None else None,
    effective_limit=effective,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
    no_device=True)))
