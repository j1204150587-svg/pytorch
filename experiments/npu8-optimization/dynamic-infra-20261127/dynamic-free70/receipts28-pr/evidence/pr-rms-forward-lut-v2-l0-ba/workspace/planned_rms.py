"""Training adapter for a single verified LUT kernel, on the current physical stream."""
from collections import Counter
import json
import os
from pathlib import Path
import threading
import time

import torch
import torch_npu
import fold_probe
import lut_rms as rms
from forward_probe import plans
from lut_kernel import lut_forward


ENABLED = os.getenv("NPU_DYNAMIC_PLANNED_RMS_LUT")=="1"
SWITCH = os.getenv("TORCHINDUCTOR_NPU_FAST_LAUNCH")=="1"
DEBUG = any(os.getenv(k,"").lower() in ("true","1","yes","on") for k in
            ("TRITON_INTERPRET","TRITON_DEVICE_PRINT","INDUCTOR_ASCEND_DUMP_FX_GRAPH",
             "INDUCTOR_ASCEND_CHECK_ACCURACY","TORCHINDUCTOR_DUMP_LAUNCH_PARAMS"))
RAW = getattr(torch_npu._C,"_npu_getCurrentRawStreamNoWait",None) or torch_npu._C._npu_getCurrentRawStream
LOCK = threading.Lock()
STATE = {}
INFO = {}
COUNTS = Counter()
SHAPES = Counter()
ORIGINAL_PARTS = rms.parts


def planned_parts(x,w,table,debug=False):
    if debug:
        return ORIGINAL_PARTS(x,w,table,debug=True)
    dev=x.device.index
    b,l,_=x.shape
    grid=b*l
    inv=x.new_empty((b,l,1))
    z=torch.empty_like(x)
    y=torch.empty_like(x)
    args=(x,w,table,inv,z,y,x)
    def jit():
        return lut_forward[(grid,)](*args,False)
    # A standard launch both registers the actual function and initializes these
    # outputs. Once it is issued, it must not be replayed through the plan.
    cached=STATE.get(dev)
    if cached is None and SWITCH and not DEBUG and not torch.autograd.profiler._is_profiler_enabled:
        with LOCK:
            cached=STATE.get(dev)
            if cached is None:
                start=time.perf_counter_ns()
                binary=jit()
                witness={}
                plan=plans(binary,witness,fold_probe)
                STATE[dev]=(binary,plan)
                INFO[dev]={"abi":witness,"first_stock_and_prepare_host_us":(time.perf_counter_ns()-start)/1000}
                COUNTS["cold_stock"]+=1
                SHAPES[b,l,"cold_stock"]+=1
                return y,inv,z,()
    reason=None
    if not SWITCH or DEBUG:
        reason="disabled_or_debug"
    elif torch.autograd.profiler._is_profiler_enabled:
        reason="profiler"
    elif cached is None or cached[1] is None:
        reason="missing_verified_plan"
    elif fold_probe.active_hooks(cached[0]):
        reason="triton_hooks"
    elif (dev!=torch.npu.current_device() or not 0<grid<=65535 or
          any(v.device!=x.device or v.dtype!=torch.bfloat16 or not v.is_contiguous() or v.data_ptr()%16 for v in args)):
        reason="geometry_or_active_device"
    else:
        raw=int(RAW(dev))
        if not raw:
            reason="missing_stream"
        elif grid>40 and not callable(cached[1][2]):
            reason="unavailable_physical_cap"
    if reason:
        jit()
        COUNTS["fallback_"+reason]+=1
        SHAPES[b,l,"fallback_"+reason]+=1
    else:
        plan,direct,capped=cached[1]
        COUNTS["entered"]+=1
        if grid>40:
            capped(plan,raw,grid,1,1,args,40)
            route="capped"
        else:
            direct(plan,raw,grid,1,1,args)
            route="direct"
        COUNTS[route]+=1
        COUNTS["returned"]+=1
        SHAPES[b,l,route]+=1
    return y,inv,z,()


def install(library):
    torch.ops.load_library(library)
    rms.parts=planned_parts


def run(x,w,eps):
    return rms.run(x,w,eps,ENABLED)


def finish(path,steps):
    actual=dict(rms.COUNTS)
    expected=9*steps
    if ENABLED:
        correct=(actual.get("triton_forward",0)==expected and actual.get("fallback",0)==0 and
                 COUNTS["cold_stock"]==1 and COUNTS["entered"]==COUNTS["returned"]==expected-1 and
                 COUNTS["direct"]+COUNTS["capped"]==expected-1 and actual.get("table_created")==1)
    else:
        correct=(actual.get("stock_forward")==expected and not COUNTS and not actual.get("table_created",0))
    Path(path).write_text(json.dumps({"enabled":ENABLED,"expected_forwards":expected,"all_expected":correct,
        "rms":actual,"tables":rms.BUILDS,"launches":dict(COUNTS),"plans":INFO,
        "shape_routes":[{"shape":[b,l],"route":route,"calls":n} for (b,l,route),n in sorted(SHAPES.items())]},indent=2)+"\n")
