"""Bind only score binaries whose actual generated driver ABI has been checked."""
from collections import Counter
import hashlib
import inspect
from pathlib import Path
import re
import threading

import torch
import torch_npu
from triton.backends.ascend import driver
import fold_probe

DRIVER="b39118053062f3fc6d7f1047f8c9eb229de47a2b6106d8119cc7087f5b38818e"
KERNEL_SOURCE="b552c38e5e52e9c0ec85a2b6e0f6ea1addefd046357362d174ba8fb6bd2a57b4"
COUNTS=Counter()
BINS={}
PLANS={}
EVIDENCE={}
CACHE_LOCK=threading.RLock()


def prepare(binary,kernel,tensors):
    ext=torch_npu._C
    name=kernel.fn.__name__
    index=lambda k: kernel.arg_names.index(k) if isinstance(k,str) else k
    dtypes=tuple("*fp32" if t.dtype==torch.float32 else "*bf16" for t in tensors)
    signature={index(k):str(v) for k,v in binary.src.signature.items()}
    meta=fold_probe.values(binary.metadata)
    if (signature!=dict(enumerate(dtypes)) or binary.src.constants or meta["name"]!=name
            or meta["parallel_mode"]!="simd" or meta["force_simt_only"]
            or int(meta.get("coalesce_factor") or 1)!=1 or int(meta.get("workspace_size") or 0)>0):
        raise ValueError(f"Unrecognized score kernel contract: {meta}")
    text=driver.generate_npu_wrapper_src({},signature,binary.metadata)
    witness=fold_probe.witness_eager_launcher(binary)
    elf,=[v for v in binary.asm.values() if isinstance(v,bytes) and v.startswith(b"\x7fELF")]
    elf_digest=hashlib.sha256(elf).hexdigest()
    init=re.search(r"}\s*args\s*=\s*{(.*?)};",text,re.S).group(1)
    locks=int(meta.get("lock_num",-1))
    value=int(meta.get("lock_init_value",meta.get("lock_init_val",0)))
    slot="static_cast<void*>(syncBlockLock_ptr)" if locks>0 else "nullptr"
    parts=["static_cast<void*>(ffts_addr)",slot,"nullptr",
           *(f"static_cast<void*>(arg{i})" for i in range(len(signature))),
           *(f"static_cast<int32_t>(grid{i})" for i in "XYZ")]
    if ([x.strip() for x in init.split(",")]!=parts or witness["driver_py_sha256"]!=DRIVER
            or witness["allocates_workspace"] or not witness["uses_ffts"] or locks not in (-1,0,8)
            or witness["allocates_sync_lock"]!=(locks==8)):
        raise ValueError(f"Unrecognized score lock ABI: {locks=} {value=} {witness} {init}")
    cap=witness["auto_map_physical_cap"]
    # SIMD rtKernelLaunch has no localMemorySize parameter. Ascend910_9362
    # reports shared_mem_dynamic_size even on this ordinary SIMD entry point.
    if (cap not in (None,40) or text.count("uint32_t blockNum = gridX * gridY * gridZ;")!=2
            or "ret = rtKernelLaunchWithFlagV2(" in text or text.count("ret = rtKernelLaunch(func, blockNum,")!=2):
        raise ValueError("Unrecognized runtime block count")
    if cap and text.count("blockNum = std::min(blockNum, (uint32_t)40);")!=2:
        raise ValueError("Entry points disagree on physical cap")
    source=Path(inspect.getsourcefile(kernel.fn)).read_bytes().replace(b"\r\n",b"\n")
    if (name not in ("causal_score_forward","causal_score_backward") or
            hashlib.sha256(source).hexdigest()!=KERNEL_SOURCE or locks!=8 or cap is not None):
        raise ValueError("Expected masked BF16 indices and the two-dimensional lock ABI")
    api=Path(driver.NPUUtils().get_so_path()).resolve(strict=True)
    if locks==8:
        if (value!=0 or text.count("syncBlockLockSize = 8 * sizeof(int64_t)")!=2
                or text.count("lockInitData(8, 0)")!=2 or text.count("RT_MEMCPY_HOST_TO_DEVICE")<2
                or str(api) not in text or "triton_release_retained_tensor" not in text):
            raise ValueError("Different lock initialization or utility")
        plan=ext._npu_hstu_make_score_lock_plan(name,binary.function,("tensor",)*len(tensors),str(api))
    else:
        plan=ext._npu_inductor_make_fast_launch_plan(name,binary.function,("tensor",)*len(tensors),
                                                    False,0,False,True)
    plan._owner=binary
    EVIDENCE[name]={"witness":witness,"initializer":init,"image_sha256":elf_digest,"signature":signature,
                    "lock_num":locks,"lock_init":value,"physical_cap":cap,"utils":str(api),
                    "launch_api":"rtKernelLaunch","reported_dynamic_local_memory":meta.get("shared_mem_dynamic_size"),
                    "utils_sha256":hashlib.sha256(api.read_bytes()).hexdigest(),"grid_argument":"[L,B,1]"}
    return plan,locks,cap


def launch(kernel,tensors,native):
    x=tensors[0]
    batch,_,length,_=x.shape
    shape=(length,batch,1)
    name=kernel.fn.__name__
    key=(name,x.device)
    saved=BINS.get(key)
    if (torch.autograd.profiler._is_profiler_enabled or
            (saved is not None and fold_probe.active_hooks(saved)) or any(t.data_ptr()%16 for t in tensors)):
        COUNTS["fallback"]+=1
        kernel[shape](*tensors)
        return
    if not native or saved is None:
        binary=kernel[shape](*tensors)
        with CACHE_LOCK:
            if key in BINS and BINS[key] is not binary:
                raise ValueError("kernel specialization changed")
            BINS[key]=binary
        COUNTS["jit_"+name]+=1
        if native:COUNTS["cold_jit"]+=1
        return
    with CACHE_LOCK:
        if key not in PLANS:PLANS[key]=prepare(saved,kernel,tensors)
        plan,locks,cap=PLANS[key]
    if torch.npu.current_device()!=x.device.index or any(t.device!=x.device for t in tensors):
        raise ValueError("mixed device")
    stream=torch.npu.current_stream(x.device)
    # An asynchronously consumed tensor may have been allocated on another
    # stream and released by its owner just after launch.
    for tensor in tensors:tensor.record_stream(stream)
    raw=torch_npu._C._npu_getCurrentRawStreamNoWait(x.device.index)
    if cap is None or batch*length<=cap:
        torch_npu._C._npu_inductor_fast_launch_with_plan(plan,raw,*shape,tensors)
    else:
        torch_npu._C._npu_hstu_fast_launch_with_physical_cap(plan,raw,*shape,tensors,cap)
    COUNTS["planned_"+name]+=1
    COUNTS["requested_lock_init_h2d"]+=int(locks>0)


def service_counts():
    return {key[0]:list(torch_npu._C._npu_hstu_score_lock_counts(plan))
            for key,(plan,locks,cap) in PLANS.items() if locks>0}
