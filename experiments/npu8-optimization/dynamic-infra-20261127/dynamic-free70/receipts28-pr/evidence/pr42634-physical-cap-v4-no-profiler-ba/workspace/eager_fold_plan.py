"""Experiment-local eager adapter for two-tensor, constexpr-1024 ordered_64."""
from collections import Counter
import json
import os
from pathlib import Path
import sys

import torch
import torch_npu
from triton.compiler.compiler import CompiledKernel
from fold_probe import prepare
from npu_ordered_fold import ordered_64


ENABLED = os.environ.get("NPU_EAGER_FOLD_PLAN") == "1"
SWITCH = os.environ.get("TORCHINDUCTOR_NPU_FAST_LAUNCH") == "1"
RAW_STREAM = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
if RAW_STREAM is None:
    RAW_STREAM = torch_npu._C._npu_getCurrentRawStream
ROWS = []
TOTAL = Counter()
INFO = {}
BINARY = None
PLAN = None
DEVICE = None
PLAN_FAILURE = None
STEP = None
try:
    from triton import knobs
    HOOK_OWNER = knobs.runtime
except ImportError:
    HOOK_OWNER = CompiledKernel


def begin(step):
    global STEP
    STEP = int(step)


def hooks(name):
    hook = getattr(HOOK_OWNER, name, None)
    if hook is None:
        return 0
    callbacks = getattr(hook, "calls", None)
    return len(callbacks) if callbacks is not None else 1


def environment_flag(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes", "on")


STATIC_GUARDS = [name for name in ("TRITON_INTERPRET", "INDUCTOR_ASCEND_DUMP_FX_GRAPH",
                 "INDUCTOR_ASCEND_CHECK_ACCURACY", "TORCHINDUCTOR_DUMP_LAUNCH_PARAMS", "TRITON_DEVICE_PRINT")
                 if environment_flag(name)]


def launch(x, out):
    global BINARY, PLAN, DEVICE, PLAN_FAILURE
    tables, batch, length, width = x.shape
    grid = batch * length
    device = x.device.index
    stream = int(RAW_STREAM(device))
    profiler = bool(torch.autograd.profiler._is_profiler_enabled)
    enter = hooks("launch_enter_hook")
    leave = hooks("launch_exit_hook")
    row = {"step": STEP, "shape": [batch,length], "grid": [grid,1,1], "stream":stream,
           "device":device, "torch_profiler":profiler, "enter_hooks":enter, "exit_hooks":leave}
    ROWS.append(row)
    reason = None
    if not ENABLED:
        reason = "stock"
    elif not SWITCH or STATIC_GUARDS:
        reason = "disabled_or_debug"
    elif profiler:
        reason = "profiler"
    elif enter or leave:
        reason = "triton_launch_hooks"
    elif (device != torch.npu.current_device() or out.device != x.device or
          tables != 1024 or width != 64 or x.dtype != out.dtype or out.dtype != torch.bfloat16
          or out.shape != (batch, length, width) or not x.is_contiguous() or not out.is_contiguous()
          or x.data_ptr() % 16 or out.data_ptr() % 16 or not stream or not (0 < grid <= 65535)):
        reason = "device_layout_alignment_or_grid"
    elif BINARY is not None and DEVICE != device:
        reason = "device_change"
    elif PLAN_FAILURE:
        reason = PLAN_FAILURE
    elif PLAN is None:
        reason = "initial_stock_launch"
    elif not INFO.get("verified_physical_cap") and INFO["driver_witness"]["auto_map_physical_cap"] is not None and grid > INFO["driver_witness"]["auto_map_physical_cap"]:
        reason = "physical_grid_cap"
    if reason:
        row["route"] = reason
        TOTAL[reason] += 1
        binary = ordered_64[(grid,)](x,out,tables)
        row["binary_hash"] = getattr(binary,"hash",None)
        if BINARY is not None and binary is not BINARY and device == DEVICE:
            PLAN = None
            PLAN_FAILURE = "observed_binary_identity_change"
        if BINARY is None and (not ENABLED or reason == "initial_stock_launch"):
            BINARY = binary
            DEVICE = device
            if ENABLED:
                ready = prepare(binary,torch_npu._C,INFO)
                if isinstance(ready,str):
                    PLAN_FAILURE = ready
                    TOTAL["plan_unavailable"] += 1
                else:
                    PLAN = ready
                    TOTAL["plan_created"] += 1
        return
    row["binary_hash"] = BINARY.hash
    row["route"] = "cpp_entered"
    TOTAL["cpp_entered"] += 1
    plan, submit = PLAN
    # Every entry past this point is potentially submitted. Propagate any error.
    submit(plan,stream,grid,1,1,(x,out))
    row["route"] = "cpp_returned"
    cap = INFO["driver_witness"]["auto_map_physical_cap"]
    row["physical_blocks"] = min(grid, cap) if cap else grid
    if cap and grid > cap:
        TOTAL["cpp_capped_returned"] += 1
    TOTAL["cpp_returned"] += 1


def finish(path):
    package = Path(torch_npu.__file__).resolve()
    result = {"adapter_enabled": ENABLED, "pr_switch": SWITCH, "static_debug_guards": STATIC_GUARDS,
              "torch_npu": str(package), "raw_stream_getter": RAW_STREAM.__name__,
              "loaded_torch_compile": "torch_npu._inductor.experimental.python_wrapper_fast_launch.bind" in sys.modules,
              "counts": dict(TOTAL), "plan_failure": PLAN_FAILURE, "plan": INFO, "steps": ROWS}
    Path(path).write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
