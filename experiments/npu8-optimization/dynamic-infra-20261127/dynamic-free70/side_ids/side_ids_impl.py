"""Guarded ordinary/verified planned launches for contiguous HSTU index preparation."""
from collections import Counter
import hashlib
import os
import re
import threading

import torch

COUNTS = Counter()
INFO = {}
CACHE = {}
LOCK = threading.Lock()


def reference(ids, offsets, rows):
    local = ids.movedim(2, 0)
    return torch.where((local >= 0) & (local < rows), local + offsets, offsets.shape[0]*rows).contiguous()


def reason(ids, offsets, rows):
    if ids.ndim != 4 or tuple(ids.shape[2:]) != (1024,4) or rows != 64 or tuple(offsets.shape) != (1024,1,1,1):
        return "geometry"
    if ids.dtype not in (torch.int32,torch.int64) or offsets.dtype != torch.int64:
        return "dtype"
    if ids.device.type != "npu" or offsets.device != ids.device or torch.npu.current_device() != ids.device.index:
        return "device"
    import torch_npu
    if torch_npu.get_npu_format(ids) not in (0,2) or torch_npu.get_npu_format(offsets) not in (0,2):
        return "non_dense_format"
    if not ids.is_contiguous() or not offsets.is_contiguous():
        return "noncontiguous"
    if ids.shape[0] < 1 or ids.shape[1] < 1 or ids.shape[0]*ids.shape[1] > 4095:
        return "grid"
    if ids.data_ptr()%16 or offsets.data_ptr()%16:
        return "alignment"
    return None


def make_plan(binary, kernel, ids):
    import side_ids_driver as fold_probe
    import torch_npu
    from triton.backends.ascend import driver
    names = list(kernel.arg_names)
    signature = {(names.index(k) if isinstance(k,str) else k): str(v) for k,v in binary.src.signature.items()}
    meta = fold_probe.values(binary.metadata)
    expected = {0:"*i64" if ids.dtype == torch.int64 else "*i32", 1:"*i64",2:"*i64"}
    if (signature != expected or binary.src.constants or meta.get("name") != kernel.fn.__name__
            or meta.get("parallel_mode") != "simd" or meta.get("force_simt_only")
            or int(meta.get("coalesce_factor") or 1) != 1):
        raise ValueError(f"unsupported kernel signature or metadata: {signature} {meta}")
    witness = fold_probe.witness_eager_launcher(binary)
    src = driver.generate_npu_wrapper_src({},signature,binary.metadata)
    init = re.search(r"}\s*args\s*=\s*{(.*?)};",src,re.S)
    if init is None:
        raise ValueError("could not identify argument initializer")
    caps = re.findall(r"blockNum\s*=\s*std::min\(blockNum,\s*\(uint32_t\)(\d+)\)",src)
    blocks = re.findall(r"uint32_t blockNum\s*=\s*gridX\s*\*\s*gridY\s*\*\s*gridZ",src)
    args = init.group(1)
    if (witness["driver_py_sha256"]!="b39118053062f3fc6d7f1047f8c9eb229de47a2b6106d8119cc7087f5b38818e"
            or not witness["uses_ffts"] or not witness["actual_null_resource_slots"]
            or witness["allocates_workspace"] or witness["allocates_sync_lock"]
            or len(caps)!=len(blocks) or set(caps)!={"40"}
            or re.findall(r"\(grid([XYZ])\)",args)!=list("XYZ")
            or re.findall(r"static_cast<void\*>\(arg(\d+)\)",args)!=["0","1","2"]):
        raise ValueError("unverified launcher layout or physical block cap")
    image, = [v for v in binary.asm.values() if isinstance(v,bytes) and v.startswith(b"\x7fELF")]
    cap=getattr(torch_npu._C,"_npu_hstu_fast_launch_with_physical_cap",None)
    if cap is None:raise ValueError("no physical-cap binding in this overlay")
    plan = torch_npu._C._npu_inductor_make_fast_launch_plan(kernel.fn.__name__,binary.function,
              ("tensor","tensor","tensor"),False,int(meta.get("shared_mem_dynamic_size") or 0),False,True)
    return plan,cap,{"signature":signature,"witness":witness,"initializer":args,
                     "binary_sha256":hashlib.sha256(image).hexdigest()}


def prepare_ids(ids, offsets, rows, route, output=None):
    if route == "A":
        COUNTS["stock"] += 1
        return reference(ids,offsets,rows)
    if route not in ("S","P"):
        raise ValueError(route)
    why=reason(ids,offsets,rows)
    if why:
        COUNTS["fallback_"+why]+=1
        return reference(ids,offsets,rows)
    import side_ids_driver as fold_probe
    import torch_npu
    from side_ids_triton import side_ids_kernel
    b,l,_,f=ids.shape
    shape=(1024,b,l,f)
    out=ids.new_empty(shape,dtype=torch.int64) if output is None else output
    if out.shape!=shape or out.device!=ids.device or out.dtype!=torch.int64 or not out.is_contiguous() or out.data_ptr()%16:
        raise ValueError("invalid preallocated output")
    grid=(b*l,16,1)
    stream=torch.npu.current_stream(ids.device)
    # The C++ task stores addresses. Keep allocator blocks tied to the actual consuming stream.
    for tensor in (ids,offsets,out):
        tensor.record_stream(stream)
    def stock_kernel():
        return side_ids_kernel[grid](ids,offsets,out)
    if route=="S":
        stock_kernel()
        COUNTS["triton"]+=1
        return out
    if (os.getenv("TORCHINDUCTOR_NPU_FAST_LAUNCH")!="1" or torch.autograd.profiler._is_profiler_enabled
            or any(os.environ.get(k,"").lower() in ("1","true","yes","on") for k in
                   ("TRITON_INTERPRET","TRITON_DEVICE_PRINT","INDUCTOR_ASCEND_DUMP_FX_GRAPH",
                    "INDUCTOR_ASCEND_CHECK_ACCURACY","TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"))):
        stock_kernel()
        COUNTS["fallback_profiler_debug_or_switch"]+=1
        return out
    key=(ids.device.index,ids.dtype)
    saved=CACHE.get(key)
    if saved is None:
        with LOCK:
            saved=CACHE.get(key)
            if saved is None:
                binary=stock_kernel()
                try:
                    plan,cap,evidence=make_plan(binary,side_ids_kernel,ids)
                except (AttributeError,RuntimeError,ValueError) as e:
                    plan,cap,evidence=None,None,{"declined":str(e)}
                CACHE[key]=(binary,plan,cap)
                INFO[str(key)]=evidence
                COUNTS["cold_triton"]+=1
                return out
    binary,plan,cap=saved
    raw=torch_npu._C._npu_getCurrentRawStreamNoWait(ids.device.index)
    if plan is None or not raw or fold_probe.active_hooks(binary):
        stock_kernel()
        COUNTS["fallback_plan_stream_hooks"]+=1
        return out
    COUNTS["cpp_entered"]+=1
    cap(plan,int(raw),*grid,(ids,offsets,out),40)
    COUNTS["cpp_returned"]+=1
    return out

def receipt(path, steps):
    import json
    route=os.getenv("NPU_DYNAMIC_SIDE_IDS_ROUTE","A")
    if route=="A":
        passed=COUNTS=={"stock":steps}
    elif route=="S":
        passed=COUNTS=={"triton":steps}
    elif route=="P":
        passed=(COUNTS["cold_triton"]==1 and COUNTS["cpp_entered"]==COUNTS["cpp_returned"]==steps-1
                and not any(k.startswith("fallback") and n for k,n in COUNTS.items()))
    else:
        raise ValueError(route)
    path.write_text(json.dumps({"route":route,"steps":steps,"counts":dict(COUNTS),"all_expected":passed,
                                "verified_abi":INFO}, indent=2)+"\n",encoding="utf8")
