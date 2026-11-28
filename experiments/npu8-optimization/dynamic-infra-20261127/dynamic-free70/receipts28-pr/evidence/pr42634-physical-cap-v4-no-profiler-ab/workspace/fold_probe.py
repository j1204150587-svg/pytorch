"""Exercise the PR's C++ plan on the already compiled, unchanged eager ordered_64."""
import argparse
from collections import Counter
import hashlib
from importlib import import_module
import json
import os
from pathlib import Path
import re
import statistics
import sys
import time

import torch
import torch_npu
import triton
from triton.compiler.compiler import CompiledKernel

# Select the target runtime's ABI once; hook registrations change at runtime.
try:
    from triton import knobs
    HOOK_OWNER = knobs.runtime
except ImportError:
    HOOK_OWNER = CompiledKernel


def values(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "_asdict"):
        return obj._asdict()
    return vars(obj)


def active_hooks(binary):
    for name in ("launch_enter_hook", "launch_exit_hook"):
        h = getattr(HOOK_OWNER, name, None)
        if h is not None and bool(getattr(h, "calls", (h,))):
            return True
    return False


def witness_eager_launcher(binary):
    """Regenerate the actual NPULauncher C++ and require its loaded content-addressed .so."""
    from triton.backends.ascend import driver
    original = binary.run
    actual = Path(original.get_launcher_so_path()).resolve()
    key = lambda arg: binary.src.fn.arg_names.index(arg) if isinstance(arg, str) else arg
    constants = {key(k): v for k,v in binary.src.constants.items()}
    signature = {key(k): v for k,v in binary.src.signature.items()}
    body = driver.generate_npu_wrapper_src(constants, signature, binary.metadata)
    header = driver.generate_npu_header_src()
    source_hash = hashlib.sha256((header + "\0" + body).encode()).hexdigest()
    found = driver.get_cache_manager(source_hash).get_file(actual.name)
    if not found or Path(found).resolve() != actual:
        raise ValueError("regenerated driver C++ cache key does not match the loaded launcher")
    caps = set(re.findall(r"blockNum\s*=\s*std::min\(blockNum,\s*\(uint32_t\)(\d+)\)", body))
    if len(caps) > 1 or ("std::min(blockNum" in body and not caps):
        raise ValueError("unresolved physical block cap")
    if "rtKernelLaunch(func, blockNum," not in body:
        raise ValueError("unrecognized baseline launch API")
    return {"original_shared_object":str(actual), "shared_object_sha256":hashlib.sha256(actual.read_bytes()).hexdigest(),
            "driver_py_sha256":hashlib.sha256(Path(driver.__file__).read_bytes()).hexdigest(),
            "launcher_source_cache_hash":source_hash,
            "allocates_workspace":bool(re.search(r"uint64_t\s+totalWorkSpaceSize\s*=", body)),
            "allocates_sync_lock":bool(re.search(r"uint64_t\s+syncBlockLockSize\s*=", body)),
            "actual_null_resource_slots":bool(re.search(r"nullptr,\s*nullptr,\s*static_cast<void\*>\(arg0\)", body)),
            "uses_ffts":"ret = rtGetC2cCtrlAddr(" in body, "auto_map_physical_cap":int(next(iter(caps))) if caps else None}


def prepare(binary, extension, report):
    # Metadata and signature belong to the result of an ordinary, successful JIT launch.
    sig = binary.src.signature
    constants = getattr(binary.src, "constexprs", None)
    if constants is None:
        constants = getattr(binary.src, "constants", None)
    metadata = values(binary.metadata)
    stub = getattr(binary, "function", None)
    if stub is None:
        stub = getattr(binary, "cu_function", None)
    report["compiled"] = {"class": type(binary).__name__, "signature": str(sig), "constants": str(constants),
                          "metadata": {str(k): str(v) for k,v in metadata.items()},
                          "has_stub": bool(stub), "hash": str(getattr(binary, "hash", ""))}
    if [str(v) for v in sig.values() if str(v) != "constexpr"] != ["*bf16", "*bf16"]:
        return "unexpected_eager_fold_ABI"
    if not isinstance(constants, dict) or not any(key in ("tables", 2, (2,)) and getattr(v, "value", v) == 1024
                                                  for key, v in constants.items()):
        return "tables_specialization_unknown"
    if not stub:
        return "missing_compiled_function"
    if any(k not in metadata for k in ("parallel_mode", "force_simt_only")):
        return "compiled_resource_metadata_unknown"
    if any(int(metadata.get(k) or 0) > 0 for k in ("workspace_size", "lock_num") if k in metadata):
        return "compiled_resource_required"
    try:
        witness = witness_eager_launcher(binary)
    except Exception as exc:
        return "stock_launcher_ABI_evidence_unavailable:" + str(exc)
    report["driver_witness"] = witness
    if witness["allocates_workspace"] or witness["allocates_sync_lock"] or not witness["actual_null_resource_slots"]:
        return "stock_launcher_requires_resources_or_slots_unresolved"
    if int(metadata.get("coalesce_factor") or 1) > 1 and int(metadata.get("coalesce_axis", -1)) in (0, 1, 2):
        return "original_launcher_coalesces_logical_grid"
    if os.getenv("TRITON_DEVICE_PRINT", "").lower() in ("1", "true"):
        return "device_print"
    backend_make = getattr(extension, "_npu_inductor_make_fast_launch_plan", None)
    backend_call = getattr(extension, "_npu_inductor_fast_launch_with_plan", None)
    if not callable(backend_make) or not callable(backend_call):
        return "planned_backend_unavailable"
    from triton.backends.ascend.utils import force_disable_ffts, get_ascend_arch_from_env, is_ffts_supported
    arch = get_ascend_arch_from_env()
    ffts = bool(is_ffts_supported(arch) and not force_disable_ffts())
    if ffts != witness["uses_ffts"]:
        return "generated_launcher_FFTS_mismatch"
    if type(metadata["force_simt_only"]) not in (bool, int) or metadata["force_simt_only"] not in (False, True):
        return "compiled_SIMT_metadata_unknown"
    force_simt = bool(metadata["force_simt_only"])
    simt = force_simt or "simt" in str(metadata["parallel_mode"]).lower()
    if simt and "shared_mem_dynamic_size" not in metadata:
        return "compiled_SIMT_shared_mem_unknown"
    report["plan_config"] = {"ffts": ffts, "arch": str(arch), "simt": simt, "force_simt": force_simt,
                             "shared_mem_dynamic_size": int(metadata.get("shared_mem_dynamic_size") or 0)}
    try:
        plan = backend_make("ordered_64", stub, ("tensor", "tensor"), simt,
                            report["plan_config"]["shared_mem_dynamic_size"], force_simt, ffts)
        plan._owner = binary
    except Exception as exc:
        return "plan_creation_error:" + str(exc)
    enabled = os.getenv("NPU_DYNAMIC_FOLD_GRID_CAP") == "1"
    known_driver = (witness["launcher_source_cache_hash"] ==
                    "911eed289b4d5b9773e4aaaef768c08cb28b28acfc8a493ca9e338bdc9a14e05"
                    and witness["auto_map_physical_cap"] == 40
                    and witness["driver_py_sha256"] ==
                    "b39118053062f3fc6d7f1047f8c9eb229de47a2b6106d8119cc7087f5b38818e")
    capped = getattr(extension, "_npu_hstu_fast_launch_with_physical_cap", None)
    report["verified_physical_cap"] = bool(enabled and known_driver and callable(capped))
    if report["verified_physical_cap"]:
        def launch(plan, stream, x, y, z, tensors):
            if x * y * z > 40:
                return capped(plan, stream, x, y, z, tensors, 40)
            return backend_call(plan, stream, x, y, z, tensors)
        return plan, launch
    return plan, backend_call


def run(args, report):
    sys.path.insert(0, str(args.fold_dir.resolve()))
    from npu_ordered_fold import ordered_64
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    extension = import_module(args.bindings_module)
    report["bindings_module"] = {"name": args.bindings_module, "file": str(getattr(extension, "__file__", ""))}
    report["environment"] = {k: os.getenv(k) for k in ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES",
                              "TASK_QUEUE_ENABLE", "TORCHINDUCTOR_NPU_FAST_LAUNCH", "TRITON_INTERPRET")}
    report["versions"] = {"torch": str(torch.__version__), "torch_npu": str(torch_npu.__version__),
                          "triton": str(triton.__version__)}
    get_stream = getattr(torch_npu._C, "_npu_getCurrentRawStreamNoWait", None)
    if get_stream is None:
        get_stream = torch_npu._C._npu_getCurrentRawStream
    report["get_stream"] = get_stream.__name__
    cpu_rng = torch.Generator().manual_seed(20261127)
    dims = [(1,19), (2,14), (2,22), (4,31), (2,14)]
    data = []
    for b, s in dims:
        cpu = torch.randn((1024,b,s,64), generator=cpu_rng).to(torch.bfloat16)
        source = cpu.npu()
        a = torch.empty((b,s,64), device=source.device, dtype=source.dtype)
        c = torch.empty_like(a)
        data.append((source, a, c))
    def stock(x, y):
        return ordered_64[(y.numel() // 64,)](x, y, 1024)
    seen = [stock(x, a) for x, a, _ in data]
    torch.npu.synchronize()
    if any(b is not seen[0] for b in seen):
        report["pre_submit_fallback"] = "compiled_binary_identity_changed"
        return
    binary = seen[0]
    if binary is None:
        report["pre_submit_fallback"] = "torch_triton_JIT_did_not_return_CompiledKernel"
        return
    direct = prepare(binary, extension, report)
    if isinstance(direct, str):
        report["pre_submit_fallback"] = direct
        return
    plan, submit = direct
    physical_cap = report["driver_witness"]["auto_map_physical_cap"]
    if report.get("verified_physical_cap"):
        physical_cap = None
    counts = Counter()
    report["all_counts"] = counts
    def invalid_layout(x,y):
        return (x.device != y.device or x.device.index != torch.npu.current_device()
                or x.shape != (1024, *y.shape) or y.shape[-1] != 64
                or x.dtype != y.dtype or y.dtype != torch.bfloat16 or not x.is_contiguous()
                or not y.is_contiguous() or x.data_ptr() % 16 or y.data_ptr() % 16)
    def selected(x, y):
        reason = ("profiler" if torch.autograd.profiler._is_profiler_enabled else
                  "triton_launch_hooks" if active_hooks(binary) else "")
        if invalid_layout(x,y):
            reason = "device_layout_or_alignment"
        grid = y.numel() // 64
        if not reason and physical_cap is not None and grid > physical_cap:
            reason = "physical_grid_cap"
        if reason:
            counts["fallback_" + reason] += 1
            stock(x, y)
            return
        stream = get_stream(x.device.index)
        if not stream or not 0 < grid <= 65535:
            counts["fallback_stream_or_grid"] += 1
            stock(x,y)
            return
        # C++ may have submitted the kernel before raising. Never replay.
        counts["cpp_entered"] += 1
        submit(plan, stream, grid, 1, 1, (x, y))
        counts["cpp_returned"] += 1
    report["cases"] = []
    second = torch.npu.Stream()
    for i, (x, a, b) in enumerate(data):
        b.fill_(float("nan"))
        torch.npu.synchronize()
        stream = torch.npu.current_stream() if i % 2 == 0 else second
        with torch.npu.stream(stream):
            planned_stream = get_stream(x.device.index)
            before = counts.copy()
            selected(x, b)
            stream.synchronize()
        av = a.cpu().view(torch.int16)
        bv = b.cpu().view(torch.int16)
        neq = int((av != bv).sum())
        report["cases"].append({"shape": list(dims[i]), "stream": int(planned_stream), "different": neq,
                                "numel": av.numel(), "source_data_ptr": int(x.data_ptr()), "routed": dict(counts - before)})
    report["gate_counts"] = dict(counts)
    planned_expected = sum(physical_cap is None or b*s <= physical_cap for b,s in dims)
    if (any(case["different"] for case in report["cases"]) or not planned_expected
            or counts["cpp_returned"] != planned_expected or counts["fallback_physical_grid_cap"] != len(dims) - planned_expected):
        report["status"] = "bitwise_or_activation_gate_failed"
        return
    before = counts.copy()
    with torch.autograd.profiler.profile(use_cpu=True):
        selected(data[0][0], data[0][2])
    torch.npu.synchronize()
    report["profiler_guard"] = dict(counts - before)
    if report["profiler_guard"] != {"fallback_profiler": 1}:
        report["status"] = "profiler_fallback_failed"
        return
    # Separate diagnostic, outside all ordered timing windows. Use one eligible
    # real shape; synchronize only at boundaries, as in the paired measurements.
    x, a, b = next(datum for datum in data if physical_cap is None or datum[2].numel()//64 <= physical_cap)
    grid = b.numel()//64
    raw = get_stream(x.device.index)
    def legacy_hook_owner_lookup():
        try:
            from triton import knobs
            return knobs.runtime
        except ImportError:
            return type(binary)
    def time_fn(fn):
        torch.npu.synchronize()
        observations = []
        for _ in range(40):
            start = time.perf_counter_ns()
            fn()
            observations.append((time.perf_counter_ns()-start)/1000)
        torch.npu.synchronize()
        return {"calls": len(observations), "p50_us": statistics.median(observations),
                "p95_us": sorted(observations)[37], "samples_us": observations}
    report["host_breakdown"] = {
        "shape":list(x.shape[1:3]), "legacy_try_import":time_fn(legacy_hook_owner_lookup),
        "cached_owner_dynamic_hooks":time_fn(lambda: active_hooks(binary)),
        "layout_device_dtype_alignment":time_fn(lambda: invalid_layout(x,b)),
        "raw_current_stream":time_fn(lambda: get_stream(x.device.index)),
        "cpp_submit_prepared_args":time_fn(lambda: submit(plan,raw,grid,1,1,(x,b))),
        "complete_eager_adapter":time_fn(lambda: selected(x,b)),
        "stock_triton":time_fn(lambda: stock(x,a))}
    if not torch.equal(a.cpu().view(torch.int16), b.cpu().view(torch.int16)):
        report["status"] = "timing_input_bitwise_failed"
        return
    report["timing"] = []
    for order in ("AB", "BA", "AB", "BA"):
        pair = {}
        for label in order:
            torch.npu.synchronize()
            host = []
            start = time.perf_counter_ns()
            for i in range(args.repeats):
                x, a, b = data[i % len(data)]
                begin = time.perf_counter_ns()
                stock(x, a) if label == "A" else selected(x, b)
                host.append((time.perf_counter_ns() - begin)/1000)
            torch.npu.synchronize()
            pair[label] = {"synced_us": (time.perf_counter_ns()-start)/1000/args.repeats,
                           "host_us": host, "host_p50_us": statistics.median(host)}
        report["timing"].append({"order":order, "legs":pair})
    report["final_counts"] = dict(counts)
    report["status"] = "passed"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--bindings-module", default="torch_npu._C")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=32)
    args = parser.parse_args()
    report = {}
    try:
        run(args, report)
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: report.get(k) for k in ("status", "pre_submit_fallback", "gate_counts", "error")}))
