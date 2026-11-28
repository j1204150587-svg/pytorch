"""Compile three op-plugin units in a CPU-only container and relink the pinned PTA build."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--builder", type=Path, default=Path("/artifacts/dynamic-infra-20261127/build_native.py"))
    parser.add_argument("--base", type=Path, default=Path("/artifacts/pta08-queued-replay-20261125-C6eCXV"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("pinned_builder", args.builder)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.SOURCES = {
        "op_plugin/ops/opapi/EmbeddingKernelNpuOpApi.cpp":
            "a36d2188732e7a1f5a701bfd640930a9d094a5c48662299a8c630227ed63a1fd",
        "op_plugin/ops/opapi/StructKernelNpuOpApi.cpp":
            "795e6eb5ee36d3057a11cf70e6f6e3835cfd12a05183022a492eeb59d68848b2",
        "op_plugin/utils/op_api_common.cpp":
            "e49dbe4a0e6709f23d8c77437addecbbcccf14996c66db2ef66b7f1acbbb17f7",
    }
    here = Path(__file__).resolve().parent
    args.installer = here / "apply.py"
    args.output.mkdir(parents=True, exist_ok=True)
    inputs = [args.builder, here / "build.py", args.installer, here / "embedding.patch",
              here / "index_scope.h", here / "index_scope.inc"]
    sources = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (args.output / "combined-inputs.json").write_text(json.dumps(sources, indent=2) + "\n")
    module.run_build(args)


if __name__ == "__main__":
    main()
