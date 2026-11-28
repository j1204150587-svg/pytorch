"""Rebuild only the pinned embedding entry and relink baseline Ninja objects."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


SOURCE = "op_plugin/ops/opapi/StructKernelNpuOpApi.cpp"
TOKEN = "embedding_dense_backward"


def load_builder(path):
    spec = importlib.util.spec_from_file_location("pinned_npu_builder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def small_unit(pinned, body):
    """Extract the entire stock entry from the verified release source."""
    head, rest = pinned.split("at::Tensor _adaptive_avg_pool2d_backward(", 1)
    start = rest.index("at::Tensor embedding_dense_backward(")
    end = rest.index("\n}\n", start) + 3
    entry = rest[start:end]
    if pinned.count(TOKEN + "(") != 2 or entry.count(TOKEN + "(") != 2:
        raise ValueError("unexpected internal uses of embedding_dense_backward")
    call = "    EXEC_NPU_CMD(aclnnEmbeddingDenseBackward, grad_output, indices,"
    headers, ns = head.split("namespace op_api {", 1)
    if entry.count(call) != 1:
        raise ValueError("pinned entry has changed")
    entry = entry.replace(call,
        "    if (repeated_embedding::submit(grad_output, indices, num_weights, padding_idx, "
        "scale_grad_by_freq, out)) {\n        return out;\n    }\n" + call)
    return headers + body + "\nnamespace op_api {" + ns + entry + "} // namespace op_api\n"


def global_entry(path):
    # Select the public function, excluding cold clones and local static data.
    symbols = subprocess.check_output(["nm", "-g", "--defined-only", str(path)], text=True)
    names = [row.split()[-1] for row in symbols.splitlines()
             if len(row.split()) == 3 and row.split()[1] == "T"
             and row.split()[2].startswith("_ZN6op_api24embedding_dense_backwardE")]
    if len(names) != 1:
        raise ValueError(f"expected one public embedding entry in {path}, found {names}")
    return names[0]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--builder", type=Path, required=True, help="existing dynamic-infra build_native.py")
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--base", type=Path)
    args = cli.parse_args()
    helper = load_builder(args.builder.resolve(strict=True))
    base = (args.base or helper.BASE).resolve(strict=True)
    checkout = base / "checkout"
    plugin = checkout / "third_party/op-plugin"
    build = checkout / "build"
    output = args.output.resolve()
    helper.check(not output.is_relative_to(base), "use an isolated build output")
    output.mkdir(parents=True, exist_ok=True)
    dest = output / "package"
    helper.check(not dest.exists(), f"package already exists: {dest}")
    stage = Path(tempfile.mkdtemp(dir=output, prefix="compile-"))
    receipt = {"status": "preparing", "stage": str(stage), "base": str(base)}
    receipt_path = output / "build-receipt.json"

    def save():
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf8")

    try:
        receipt["pta_sha"] = helper.output(["git", "rev-parse", "HEAD"], cwd=checkout)
        receipt["op_plugin_sha"] = helper.output(["git", "rev-parse", "HEAD"], cwd=plugin)
        helper.check((receipt["pta_sha"], receipt["op_plugin_sha"]) == (helper.PTA_SHA, helper.PLUGIN_SHA),
                     "source commits differ from pinned build")
        src = plugin / SOURCE
        helper.check(helper.digest(src) == helper.SOURCES[SOURCE], "unexpected stock embedding source")
        body = Path(__file__).with_name("index_scope.inc")
        generated = stage / SOURCE
        generated.parent.mkdir(parents=True)
        generated.write_text(small_unit(src.read_text(), body.read_text()), encoding="utf8")
        db_path = build / "compile_commands.json"
        unit = helper.compile_unit(json.loads(db_path.read_text()), plugin, stage, SOURCE)
        stock = Path(unit["original_object"])
        embedded = Path(unit["object"])
        local = stage / "StockStructKernel.o"
        symbol = global_entry(stock)
        # All other ~1k operators use their original machine code. The stock
        # embedding symbol becomes private in this object; the new TU defines it.
        localize = ["objcopy", "--localize-symbol=" + symbol, str(stock), str(local)]
        subprocess.run(localize, check=True)
        so = stage / "libtorch_npu.so"
        link = helper.relink_cmd(build, [dict(unit, object=str(local))], so)
        link.insert(link.index(str(local)) + 1, str(embedded))
        receipt.update(status="compiling", compile=[unit], link=link, link_cwd=str(build), localize=localize,
                       stock_object_sha256=helper.digest(stock), small_source_sha256=helper.digest(generated),
                       compile_commands_sha256=helper.digest(db_path), input_files_sha256={
                           p.name: helper.digest(p) for p in (Path(__file__), args.builder, body, src)})
        save()
        subprocess.run(unit["argv"], cwd=unit["cwd"], check=True)
        helper.check(global_entry(embedded) == symbol, "entry ABI differs across the two objects")
        unit["object_sha256"] = helper.digest(embedded)
        receipt.update(status="linking", localized_object_sha256=helper.digest(local))
        save()
        subprocess.run(link, cwd=build, check=True)
        original = base / "package"
        relative = Path("torch_npu/lib/libtorch_npu.so")
        origin_so = original / relative
        new_package = stage / "package"
        shutil.copytree(original, new_package, symlinks=True, ignore=lambda path, names:
                        {relative.name} if Path(path) == origin_so.parent else set())
        new_so = new_package / relative
        helper.check(new_so.parent.resolve().is_relative_to(new_package), "library path escapes isolated package")
        os.replace(so, new_so)
        new_package.rename(dest)
        receipt.update(status="built", library=str(dest / relative), pythonpath=str(dest),
                       library_sha256=helper.digest(dest / relative),
                       original_package_library_sha256=helper.digest(origin_so))
        save()
        print(json.dumps({k: receipt[k] for k in ("pythonpath", "library", "library_sha256")}))
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    main()
