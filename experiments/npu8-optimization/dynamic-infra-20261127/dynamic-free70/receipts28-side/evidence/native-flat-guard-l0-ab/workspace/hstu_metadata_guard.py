"""Use an in-process C++ metadata scan, with the original Python predicate as fallback."""
import atexit
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode


def load_native(directory):
    receipt=json.loads((directory/"build.json").read_text(encoding="utf-8"))
    if torch.__version__!=receipt["torch"] or sys.version!=receipt["python"]:
        raise RuntimeError("Host guard built against a different PyTorch/Python runtime")
    binary=directory/receipt["library"]
    if hashlib.sha256(binary.read_bytes()).hexdigest()!=receipt["library_sha256"]:
        raise RuntimeError("Host guard binary differs from build receipt")
    spec=importlib.util.spec_from_file_location("_hstu_host_guard",binary)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod,receipt["library_sha256"]


class MetadataGuard:
    def __init__(self,shapes,sizes,rank_dir=None,*,enabled=None,directory=None):
        self.enabled=os.environ.get("NPU_DYNAMIC_NATIVE_FLAT_GUARD")=="1" if enabled is None else enabled
        self.plan=None
        sha=None
        if self.enabled:
            module,sha=load_native(Path(directory or os.environ["NPU_HSTU_GUARD_DIR"]))
            self.plan=module.Plan(shapes,sizes)
        self.record={"enabled":self.enabled,"binary_sha256":sha,"calls":0,"native_accept":0,
                     "native_decline":0,"context_fallback":0,"python_calls":0,"python_accept":0}
        if rank_dir is not None:
            atexit.register(self.save,Path(rank_dir))

    def evaluate(self,flat,shards,original):
        r=self.record
        r["calls"]+=1
        marker="PTAFlatGuardNative" if self.enabled else "PTAFlatGuardPython"
        with torch.autograd.profiler.record_function(marker):
            if self.enabled:
                if (torch._C._is_torch_function_mode_enabled() or _get_current_dispatch_mode() is not None or
                        torch._C._are_functorch_transforms_active() or torch.jit.is_tracing() or torch.compiler.is_compiling()):
                    r["context_fallback"]+=1
                elif self.plan.matches(flat,shards):
                    r["native_accept"]+=1
                    return flat
                else:
                    r["native_decline"]+=1
            r["python_calls"]+=1
            result=original(flat,shards)
            r["python_accept"]+=int(result is not None)
            return result

    def save(self,rank):
        r=dict(self.record)
        evidence=rank/"flat_stats_counts.json"
        stats=json.loads(evidence.read_text()) if evidence.is_file() else {}
        expected=stats.get("overflow",0)+stats.get("norm",0)
        r["flat_stats_calls"]=expected
        r["all_expected_accepted"]=(expected>0 and r["calls"]==expected and
            ((r["native_accept"]==expected and r["python_calls"]==0) if self.enabled else
             (r["python_calls"]==r["python_accept"]==expected)))
        (rank/"native_flat_guard.json").write_text(json.dumps(r,indent=2))
