"""Verify the installed, pure-pointer Triton launcher before using the PR binding."""
from collections import Counter
import hashlib
import re

import torch
import torch_npu
from triton.backends.ascend import driver
import fold_probe


class Launcher:
    def __init__(self, kernel, pointers):
        self.kernel = kernel
        self.pointers = pointers
        self.plans = {}
        self.evidence = {}
        self.calls = Counter()

    def prepare(self, binary):
        meta = fold_probe.values(binary.metadata)
        name = self.kernel.fn.__name__
        names = list(self.kernel.arg_names)
        index = lambda key: names.index(key) if isinstance(key, str) else key
        signature = {index(k): str(v) for k, v in binary.src.signature.items()}
        if (meta["name"] != name or name not in ("rms_grad_products", "rms_grad_variance")
                or signature != {i: "*bf16" for i in range(self.pointers)} or binary.src.constants
                or meta["parallel_mode"] != "simd" or meta["force_simt_only"]
                or int(meta.get("coalesce_factor") or 1) != 1):
            raise ValueError(f"unsupported kernel metadata for {name}: {meta} {signature} {binary.src.constants}")
        witness = fold_probe.witness_eager_launcher(binary)
        source = driver.generate_npu_wrapper_src({}, signature, binary.metadata)
        init = re.search(r"}\s*args\s*=\s*{(.*?)};", source, re.S).group(1)
        caps = re.findall(r"blockNum\s*=\s*std::min\(blockNum,\s*\(uint32_t\)(\d+)\)", source)
        products = re.findall(r"uint32_t blockNum\s*=\s*gridX\s*\*\s*gridY\s*\*\s*gridZ", source)
        if (witness["driver_py_sha256"] != "b39118053062f3fc6d7f1047f8c9eb229de47a2b6106d8119cc7087f5b38818e"
                or witness["auto_map_physical_cap"] != 40 or not witness["uses_ffts"]
                or witness["allocates_workspace"] or witness["allocates_sync_lock"]
                or not witness["actual_null_resource_slots"] or set(caps) != {"40"}
                or len(products) != len(caps) or re.findall(r"\(grid([XYZ])\)", init) != list("XYZ")
                or re.findall(r"static_cast<void\*>\(arg(\d+)\)", init) != list(map(str, range(self.pointers)))):
            raise ValueError(f"unexpected installed launch ABI: {witness} {init}")
        elf, = [val for val in binary.asm.values() if isinstance(val, bytes) and val.startswith(b"\x7fELF")]
        self.evidence.update(driver=witness, initializer=init, binary_sha256=hashlib.sha256(elf).hexdigest())
        return torch_npu._C._npu_inductor_make_fast_launch_plan(
            name, binary.function, ("tensor",) * self.pointers, False,
            int(meta.get("shared_mem_dynamic_size") or 0), False, True)

    def __call__(self, tensors, rows, native):
        # The first real JIT launch supplies the loaded owner and actual ABI.
        key = (tensors[0].device.index, tuple(t.data_ptr() % 16 for t in tensors))
        saved = self.plans.get(key)
        if (not native or not saved or any(key[1]) or torch.autograd.profiler._is_profiler_enabled
                or fold_probe.active_hooks(saved[0]) or not 0 < rows <= 65535):
            binary = self.kernel[(rows,)](*tensors)
            self.calls["jit"] += 1
            if native and not saved and not any(key[1]):
                self.plans[key] = (binary, self.prepare(binary))
                self.calls["prepared"] += 1
            return
        binary, plan = saved
        dev, _ = key
        if torch.npu.current_device() != dev or any(t.device != tensors[0].device for t in tensors):
            raise RuntimeError("unexpected device in NPU launch")
        stream = torch_npu._C._npu_getCurrentRawStreamNoWait(dev)
        if rows > 40:
            torch_npu._C._npu_hstu_fast_launch_with_physical_cap(plan, stream, rows, 1, 1, tensors, 40)
            self.calls["capped"] += 1
        else:
            torch_npu._C._npu_inductor_fast_launch_with_plan(plan, stream, rows, 1, 1, tensors)
            self.calls["direct"] += 1
