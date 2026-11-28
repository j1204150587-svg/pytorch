"""Choose the view-preparation strategy after the original IPG guard succeeds."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import torch
from torch.utils._python_dispatch import _get_current_dispatch_mode


def load(directory):
    data=json.loads((directory/"build.json").read_text(encoding="utf8"))
    library=directory/data["library"]
    if (torch.__version__!=data["torch"] or sys.version!=data["python"] or
        hashlib.sha256(library.read_bytes()).hexdigest()!=data["library_sha256"]):
        raise RuntimeError("Host views library or interpreter differs from the build receipt")
    spec=importlib.util.spec_from_file_location("_hstu_ipg_host_views",library)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, data["library_sha256"]


class HostViews:
    def __init__(self):
        self.mode=os.environ["NPU_IPG_HOST_VIEWS"]
        if self.mode not in ("python","native"):
            raise ValueError(self.mode)
        self.module,self.sha=load(Path(os.environ["NPU_IPG_HOST_VIEWS_DIR"]))
        self.calls=self.prepared=self.published=self.declined=self.context_fallback=0

    def prepare(self,dst,views,flags):
        self.calls+=1
        if self.mode=="python":
            return None
        if (torch._C._is_torch_function_mode_enabled() or _get_current_dispatch_mode() is not None or
            torch.is_inference_mode_enabled() or torch.compiler.is_compiling()):
            self.context_fallback+=1
            return None
        plan=self.module.prepare(dst,views,flags)
        if plan is None:
            self.declined+=1
        else:
            self.prepared+=1
        return plan

    def finish(self,rank):
        stats={name:getattr(self,name) for name in
               ("mode","sha","calls","prepared","published","declined","context_fallback")}
        (Path(rank)/"ipg_host_views.json").write_text(json.dumps(stats,indent=2)+"\n",encoding="utf8")
