"""Stage the archived #44 patch and original relinker into a new isolated build directory."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
BASE = Path("/artifacts/pta08-queued-replay-20261125-C6eCXV")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base", type=Path, default=BASE)
    args = parser.parse_args()
    out = args.output.resolve()
    base = args.base.resolve(strict=True)
    if out.is_relative_to(base):
        raise ValueError("choose a separate output for the new package")
    out.mkdir(parents=True, exist_ok=False)
    baseline_package = base / "package"
    original_lib = baseline_package / "torch_npu/lib/libtorch_npu.so"
    receipt = {}
    for name, sha in (("from-stock.patch", "7cd26cdd62bafdb684df58760607e5aec42eb41502cbf03ea3d69d2db0ec0166"),
                      ("relink.py", "6f32d42ef1f13144344a1e73f9c96f51b358b2107f29a6bb7f36519ce2b9fde5")):
        src = HERE / name
        actual = hashlib.sha256(src.read_bytes()).hexdigest()
        if actual != sha:
            raise ValueError(f"archived bytes differ for {name}: {actual}")
        shutil.copyfile(src, out / name)
        receipt[name] = actual
    (out / "inputs-sha256.json").write_text(json.dumps(receipt, indent=2) + "\n")
    shutil.copytree(baseline_package, out / "package", symlinks=True,
                    ignore=lambda path, _: {original_lib.name} if Path(path) == original_lib.parent else set())
    target_lib = out / "package/torch_npu/lib/libtorch_npu.so"
    if not target_lib.parent.resolve().is_relative_to(out) or not original_lib.is_file():
        raise ValueError("unexpected staging package")
    subprocess.run([sys.executable, "-B", str(out / "relink.py"), str(base), str(out / "from-stock.patch")],
                   cwd=out, check=True)
    print(out / "build-receipt.json", flush=True)


if __name__ == "__main__":
    main()
