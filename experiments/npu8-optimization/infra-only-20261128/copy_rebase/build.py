"""Incremental CPU-only PTA build against the verified dual embedding package."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile


def digest(path):
    with path.open("rb") as src:
        return hashlib.file_digest(src, "sha256").hexdigest()


def replaced(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"unexpected source anchor {old!r}")
    return text.replace(old, new)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, default=Path("/artifacts/infra-only-20261128/embedding/build/build-receipt.json"))
    p.add_argument("--builder", type=Path, default=Path("/artifacts/dynamic-infra-20261127/build_native.py"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--residual128", action="store_true")
    p.add_argument("--addresses", action="store_true")
    args = p.parse_args()
    parent = json.loads(args.parent.read_bytes())
    expected_parent = "4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"
    assert parent["status"] == "built" and parent["library_sha256"] == digest(Path(parent["library"])) == expected_parent
    spec = importlib.util.spec_from_file_location("build_native", args.builder)
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    previous = Path(parent["stage"])
    checkout = Path(parent["base"]) / "checkout"
    build = checkout / "build"
    plugin = checkout / "third_party/op-plugin"
    db = json.loads((build / "compile_commands.json").read_bytes())
    args.output.mkdir(exist_ok=False, parents=True)
    staged = Path(tempfile.mkdtemp(prefix="compile-", dir=args.output))
    common_name = "op_plugin/utils/op_api_common.cpp"
    caller_name = "torch_npu/csrc/aten/ops/op_api/CopyKernelOpApi.cpp"
    header_name = "op_plugin/utils/dynamic_embedding_scope.h"
    inc_name = "op_plugin/utils/dynamic_embedding_scope.inc"
    for name in (common_name, header_name, inc_name):
        assert digest(previous / name) == parent["patched_sources_sha256"][name]
    assert digest(checkout / caller_name) == "a71b330cdd1e1955592ea8b7a18d18fd418b034cb6fbfe998fa0c82455c21074"
    for name, root in ((common_name, previous), (caller_name, checkout), (header_name, previous), (inc_name, previous)):
        source = (root / name).read_text()
        if name == caller_name:
            inc = '#include "third_party/op-plugin/op_plugin/utils/op_api_common.h"'
            source = replaced(source, inc, inc + '\n#include "op_plugin/utils/dynamic_embedding_scope.h"')
            call = "    EXEC_NPU_CMD(aclnnInplaceCopy, dst, src);"
            source = replaced(source, call, "    dynamic_embedding_rebase::Scope scope(dynamic_embedding_rebase::ExactCopy{}, dst, src);\n" + call)
        elif name == header_name:
            source = replaced(source, "struct Counters;", "struct Counters;\nstruct ExactCopy {};")
            line = "    explicit Scope(const at::Tensor &indices);"
            source = replaced(source, line, line + "\n    Scope(ExactCopy, const at::Tensor &dst, const at::Tensor &src);")
        elif name == inc_name:
            include = "copy_scope_128.inc" if args.residual128 else "copy_scope.inc"
            copy_code = Path(__file__).with_name(include).read_text()
            if args.diagnostic:
                source += "\n" + Path(__file__).with_name("guard_census.inc").read_text()
                key = '    if (enabled) {\n        if (eligible_copy(dst, src, views_[0], views_[1])) {'
                copy_code = replaced(copy_code, key, key.replace("if (eligible_copy", "diagnose_copy(dst, src);\n        if (eligible_copy"))
            if args.addresses:
                source += "\n" + Path(__file__).with_name("address_census.inc").read_text()
                key = '    static const bool enabled = flag("NPU_INPLACE_COPY_REBASE");'
                copy_code = replaced(copy_code, key, "    observe_addresses(dst, src);\n" + key)
            source += "\n" + copy_code
        path = staged / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    units = [builder.compile_unit(db, plugin, staged, common_name),
             builder.compile_unit(db, checkout, staged, caller_name)]
    replacements = {Path(row["original_object"]): row["object"] for row in units}
    # Parent link already uses the compiled dual-rebase common object.
    for row in parent["compile"]:
        assert digest(Path(row["object"])) == row["object_sha256"]
        if row["source"] == common_name:
            replacements[Path(row["object"])] = replacements.pop(Path(row["original_object"]))
    link = list(parent["link"])
    lib = args.output / "libtorch_npu.so"
    builder.replace_flag(link, "-o", lib)
    for old, new in replacements.items():
        matches = [i for i, arg in enumerate(link) if arg.endswith(".o") and (build / arg).resolve() == old]
        if len(matches) != 1:
            raise ValueError(f"cannot substitute {old}, got {matches}")
        link[matches[0]] = new
    record = dict(status="compiling", parent_sha256=expected_parent, parent_receipt=str(args.parent),
                  diagnostic=args.diagnostic, residual128=args.residual128, addresses=args.addresses,
                  compile=units, link=link, staged=str(staged), patched_sources={name: digest(staged / name) for name in
                  (caller_name, common_name, header_name, inc_name)})
    receipt = args.output / "build-receipt.json"
    receipt.write_text(json.dumps(record, indent=2))
    def compile_one(row):
        with (staged / (Path(row["source"]).stem + ".compile.log")).open("wb") as log:
            subprocess.run(row["argv"], cwd=row["cwd"], stdout=log, stderr=subprocess.STDOUT, check=True)
        row["object_sha256"] = digest(Path(row["object"]))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(compile_one, units))
    subprocess.run(link, cwd=build, check=True)
    record.update(status="built", library=str(lib), library_sha256=digest(lib))
    receipt.write_text(json.dumps(record, indent=2) + "\n")
    print(record["library_sha256"], lib)


if __name__ == "__main__":
    main()
