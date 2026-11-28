"""Select one implementation of the side-slab live parameter/hook guard."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

import torch


def load(directory):
    receipt = json.loads((directory / "build.json").read_text(encoding="utf-8"))
    binary = directory / receipt["library"]
    if sys.version != receipt["python"] or hashlib.sha256(binary.read_bytes()).hexdigest() != receipt["library_sha256"]:
        raise RuntimeError("slab guard build receipt does not match this binary/interpreter")
    spec = importlib.util.spec_from_file_location("_hstu_slab_guard", binary)
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    return native


class SlabParameterGuard:
    def __init__(self, stats, *, native=None, directory=None):
        self.stats = stats
        self.enabled = os.getenv("NPU_DYNAMIC_NATIVE_SLAB_GUARD") == "1" if native is None else native
        self.module = load(Path(directory or os.environ["NPU_HSTU_SLAB_GUARD_DIR"])) if self.enabled else None
        stats.update(guard_enabled=int(self.enabled), guard_native=0, guard_python=0, guard_rejected=0,
                     guard_accepted=0, guard_shape_fallback=0)

    def matches(self, owner, tables, original):
        marker = "PTASlabParameterGuardNative" if self.enabled else "PTASlabParameterGuardPython"
        with torch.autograd.profiler.record_function(marker):
            if (self.enabled and type(tables) is torch.nn.ModuleList
                    and type(owner.tables) is tuple and type(owner.weights) is tuple and type(owner.pointers) is tuple
                    and len(owner.tables) == len(owner.weights) == len(owner.pointers)):
                self.stats["guard_native"] += 1
                result = self.module.matches(tables, owner.tables, owner.weights, owner.pointers)
            else:
                self.stats["guard_shape_fallback"] += int(self.enabled)
                self.stats["guard_python"] += 1
                result = original(tables, owner.tables, owner.weights, owner.pointers)
            self.stats["guard_accepted" if result else "guard_rejected"] += 1
            return result
