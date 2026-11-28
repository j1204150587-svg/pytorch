"""Add an experimental physical-block override to a new copy of the real PR binding."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


CPP = "23290b574bb850929a14f279e5764e6e63595574"


def build(base, out):
    info = json.loads((base / "receipt.json").read_bytes())
    args = info["command"].copy()
    stock, = [Path(x) for x in args if x.endswith("/bindings.cpp")]
    raw = stock.read_bytes()
    if hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != CPP:
        raise ValueError("unexpected PR binding source")
    src = raw.decode()
    anchor = "} // namespace\n\nvoid RegisterNPUFastLaunchBindings"
    end = '      py::arg("grid_2"),\n      py::arg("args"));\n}'
    if src.count(anchor) != 1 or src.count(end) != 1:
        raise ValueError("unrecognized PR entry")
    function = """void FastLaunchWithPhysicalCap(
    const std::shared_ptr<FastLaunchPlan>& plan, uint64_t stream,
    uint32_t x, uint32_t y, uint32_t z, const py::sequence& args, uint32_t cap) {
  TORCH_CHECK(plan != nullptr && cap > 0, "expected a plan and a positive physical cap");
  // Pack the original logical grid into the kernel argument buffer. Only
  // the independent runtime block count is capped, exactly as in NPULauncher.
  auto packed = PackLaunch(*plan, stream, x, y, z, args);
  packed.blockNum = std::min(packed.blockNum, cap);
  SubmitLaunch(*plan, std::move(packed));
}

"""
    src = '#include <algorithm>\n' + src.replace(anchor, function + anchor).replace(end, end[:-1] +
        '  m.def("_npu_hstu_fast_launch_with_physical_cap", &FastLaunchWithPhysicalCap);\n}')
    out.mkdir(parents=True, exist_ok=True)
    if (out / "package").exists() or out.resolve().is_relative_to(base.resolve()):
        raise ValueError("fresh independent output required")
    stage = Path(tempfile.mkdtemp(dir=out))
    patched = stage / "bindings.cpp"
    patched.write_text(src)
    args[args.index(str(stock))] = str(patched)
    k = args.index("-o") + 1
    name = Path(args[k]).name
    binary = stage / name
    args[k] = str(binary)
    subprocess.run(args, check=True)
    package = stage / "package"
    def copy(a,b):
        try:
            os.link(a,b)
        except OSError:
            shutil.copy2(a,b)
        return b
    shutil.copytree(base / "package", package, symlinks=True, copy_function=copy)
    target = package / "torch_npu" / name
    if not target.is_file():
        raise ValueError("PR library not found in isolated package")
    os.replace(binary, target)
    package.rename(out / "package")
    receipt = {"pr_head": info["head"], "pr_binding_blob": CPP, "command": args,
               "patched_source_sha256": hashlib.sha256(patched.read_bytes()).hexdigest(),
               "shared_library_sha256": hashlib.sha256((out/"package/torch_npu"/name).read_bytes()).hexdigest(),
               "base_binding_sha256": info["binding_module_sha256"]}
    (out/"build.json").write_text(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser()
    cli.add_argument("base", type=Path)
    cli.add_argument("out", type=Path)
    a = cli.parse_args()
    build(a.base, a.out)
