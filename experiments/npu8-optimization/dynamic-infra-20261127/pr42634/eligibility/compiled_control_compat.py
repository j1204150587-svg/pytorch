"""Exercise an actual dynamic Inductor Python wrapper independently of HSTU."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import traceback


SHAPES = ((1, 19), (2, 14), (2, 22), (4, 31))


def files_under(cache):
    output = []
    for p in sorted(cache.rglob("*.py")):
        raw = p.read_bytes()
        if b"def call(" not in raw or b"triton" not in raw:
            continue
        lines = raw.decode("utf8", errors="replace").splitlines()
        output.append(dict(path=str(p), sha256=hashlib.sha256(raw).hexdigest(),
                           lines=[line.strip() for line in lines if
                                  "bind_python_wrapper_kernel_fast" in line or
                                  ".run(" in line or "_npu_fast_launch_call_" in line][:30]))
    return output


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--profile", choices=["level0", "level1", "torch-cpu", "none"], default="level0")
    args = cli.parse_args()
    cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
    if cache.exists() and any(cache.iterdir()):
        raise ValueError("provide a fresh control-specific TORCHINDUCTOR_CACHE_DIR")
    cache.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import torch
    import torch_npu
    from fast_launch_receipt import Receipt
    torch.npu.set_device(0)
    receipt = Receipt()
    from compiled_runtime_compat import install as install_compiler_guard
    print("PR compiler guard", install_compiler_guard(), flush=True)
    shapes = []
    correctness = []
    error = None
    try:
        def pointwise(x, y):
            return x * 2.0 + y
        compiled = torch.compile(pointwise, backend="inductor", dynamic=True, fullgraph=True)
        # Integers in a small range are exactly representable in BF16 throughout.
        batches = {}
        for b, n in SHAPES:
            v = torch.arange(b*n*64).reshape(b, n, 64)
            x, y = (v % 7 - 3).bfloat16(), ((v // 3) % 9 - 4).bfloat16()
            batches[b, n] = (x.npu(), y.npu(), pointwise(x, y))

        def run(phase):
            for shape in SHAPES:
                receipt.begin(len(shapes), shape, phase)
                x, y, expected = batches[shape]
                marker = f"PR42634/compiled_control/{shape[0]}x{shape[1]}"
                marked = phase in ("level0", "level1", "torch-cpu")
                with torch.profiler.record_function(marker) if marked else nullcontext():
                    actual = compiled(x, y)
                    # Attribute this call's device work before the CPU comparison.
                    torch.npu.synchronize()
                correctness.append(torch.equal(actual.cpu(), expected))
                shapes.append(shape)

        run("first")
        run("revisit")
        if args.profile != "none":
            if args.profile == "torch-cpu":
                profile = torch.autograd.profiler.profile(use_cpu=True)
            else:
                level = (torch_npu.profiler.ProfilerLevel.Level0 if args.profile == "level0"
                         else torch_npu.profiler.ProfilerLevel.Level1)
                profile = torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
                    experimental_config=torch_npu.profiler._ExperimentalConfig(profiler_level=level),
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(args.output.parent / "profile")))
            with profile:
                run(args.profile)
        run("after")
    except Exception:
        error = traceback.format_exc()
    wrappers = files_under(cache)
    record = receipt.finish(args.output, control="independent_compiled_pointwise", error=error, wrappers=wrappers,
                            exact=correctness, cache=str(cache), profile_requested=args.profile)
    off = os.environ.get("TORCHINDUCTOR_NPU_FAST_LAUNCH", "0") == "0"
    steady = [r for r in receipt.rows if r["phase"] in ("revisit", "after")]
    verified = (error is None and all(correctness) and len(steady)==8 and bool(wrappers)
                and all(record["symbols"].values())
                and (all(not r["counts"]["planned_submit_ok"] for r in steady) if off
                     else all(r["counts"]["planned_submit_ok"] > 0 for r in steady)))
    print(json.dumps({"receipt": str(args.output), "verified": verified, "counts": record["counts"],
                      "error": error}))
    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
