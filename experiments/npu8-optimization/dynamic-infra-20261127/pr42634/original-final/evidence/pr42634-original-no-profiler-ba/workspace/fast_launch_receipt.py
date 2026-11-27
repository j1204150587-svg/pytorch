"""Count actual !42634 entry points in a diagnostic process, without loading Inductor."""
import functools
import importlib.abc
import importlib.machinery
import json
import os
import sys
from collections import Counter
from pathlib import Path


BIND = "torch_npu._inductor.experimental.python_wrapper_fast_launch.bind"
CONFIG = "torch_npu._inductor.config"
PATCH = BIND.rsplit(".", 1)[0] + ".patch"
MAKE = "_npu_inductor_make_fast_launch_plan"
LAUNCH = "_npu_inductor_fast_launch_with_plan"


class Receipt:
    def __init__(self):
        import torch
        import torch_npu
        self.profiler = torch.autograd.profiler
        self.extension = torch_npu._C
        self.counts = Counter()
        self.rows = []
        self.bind_observed = False
        self.path = None
        self.step = None
        self.dirname = str(Path(torch_npu.__file__).resolve().parent)
        self.symbols = {name: callable(getattr(self.extension, name, None)) for name in (MAKE, LAUNCH)}
        for name, label in ((MAKE, "make_plan"), (LAUNCH, "planned_submit")):
            if self.symbols[name]:
                self.wrap_native(name, label)
        if BIND in sys.modules:
            self.observe_bind(sys.modules[BIND])
        else:
            sys.meta_path.insert(0, BindFinder(self))

    def count(self, label):
        self.counts[label] += 1
        if self.step is not None:
            self.step["counts"][label] += 1

    def wrap_native(self, name, label):
        original = getattr(self.extension, name)
        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            self.count(label + "_attempt")
            try:
                result = original(*args, **kwargs)
            except Exception:
                self.count(label + "_error")
                raise
            self.count(label + "_ok")
            return result
        setattr(self.extension, name, wrapped)

    def observe_bind(self, module):
        if self.bind_observed:
            return
        bind = module.bind_python_wrapper_kernel_fast
        @functools.wraps(bind)
        def counted_bind(metadata, autotuner, **kwargs):
            grouped = module._is_grouped_autotuner(autotuner)
            self.count("grouped_bind" if grouped else "ordinary_bind")
            return bind(metadata, autotuner, **kwargs)
        module.bind_python_wrapper_kernel_fast = counted_bind

        cls = module.BoundFastLaunch
        call = cls.__call__
        @functools.wraps(call)
        def counted_call(obj, *args, **kwargs):
            self.count("bound_call")
            return call(obj, *args, **kwargs)
        cls.__call__ = counted_call
        fallback = cls._fallback
        @functools.wraps(fallback)
        def counted_fallback(obj, args, *, stream, benchmark_run, kwargs):
            # Same precedence as BoundFastLaunch.__call__. A failure during
            # promotion or launch reaches this function with no full-entry flag.
            reason = obj._static_full_entry_reason
            if reason is None and benchmark_run:
                reason = "benchmark_run"
            if reason is None and kwargs:
                reason = "runtime_kwargs"
            if reason is None and self.profiler._is_profiler_enabled:
                reason = "profiler"
            self.count("original_" + (reason or "lifecycle_or_runtime"))
            return fallback(obj, args, stream=stream, benchmark_run=benchmark_run, kwargs=kwargs)
        cls._fallback = counted_fallback
        for name, label in (("_hook_fallback", "hook_fallback"), ("_negative_fallback", "negative_fallback")):
            self.wrap_method(cls, name, label)
        self.bind_observed = True

    def wrap_method(self, owner, name, label):
        original = getattr(owner, name)
        @functools.wraps(original)
        def counted(*args, **kwargs):
            self.count(label)
            return original(*args, **kwargs)
        setattr(owner, name, counted)

    def begin(self, step, shape, phase="training"):
        self.step = {"step": step, "shape": list(shape), "phase": phase,
                     "torch_profiler_flag": bool(self.profiler._is_profiler_enabled), "counts": Counter()}
        self.rows.append(self.step)

    def finish(self, path, **extra):
        cfg = sys.modules.get(CONFIG)
        patch = sys.modules.get(PATCH)
        data = dict(switch=os.environ.get("TORCHINDUCTOR_NPU_FAST_LAUNCH"), package=self.dirname,
                    extension_file=str(getattr(self.extension, "__file__", "")), symbols=self.symbols,
                    config_loaded=cfg is not None, config_enabled=getattr(cfg, "enable_fast_launch", None),
                    patch_loaded=patch is not None, patch_installed=getattr(patch, "_PATCHED", None),
                    bind_module_observed=self.bind_observed, counts=dict(self.counts), steps=self.rows, **extra)
        dest = Path(path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return data


class BindFinder(importlib.abc.MetaPathFinder):
    """Invoke the observer when and only when Python requests the real bind module."""
    def __init__(self, receipt):
        self.receipt = receipt

    def find_spec(self, fullname, path, target=None):
        if fullname != BIND:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        original = spec.loader
        receipt = self.receipt
        class Loader:
            def create_module(self, spec):
                return original.create_module(spec)

            def exec_module(self, module):
                original.exec_module(module)
                receipt.observe_bind(module)
        spec.loader = Loader()
        return spec
