"""Recompile two op-plugin units and link an isolated TorchNPU package using the pinned Ninja build."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile


BASE = Path("/artifacts/pta08-queued-replay-20261125-C6eCXV")
PTA_SHA = "38e483ee79ec8a3286517808740a811c0f0ae66c"
PLUGIN_SHA = "eaf6b55812a1998a4a40e485e3b2861a40e9c2f6"
TARGET = "packages/torch_npu/lib/libtorch_npu.so"
SOURCES = {"op_plugin/ops/opapi/StructKernelNpuOpApi.cpp": "795e6eb5ee36d3057a11cf70e6f6e3835cfd12a05183022a492eeb59d68848b2",
           "op_plugin/utils/op_api_common.cpp": "e49dbe4a0e6709f23d8c77437addecbbcccf14996c66db2ef66b7f1acbbb17f7"}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def output(argv, cwd=None):
    return subprocess.check_output(argv, cwd=cwd, text=True).strip()


def replace_flag(argv, flag, value):
    check(argv.count(flag) == 1, f"expected one {flag} in tool command")
    idx = argv.index(flag) + 1
    old = argv[idx]
    argv[idx] = str(value)
    return old


def compile_unit(db, plugin, staged, name):
    source = plugin / name
    matches = [r for r in db if (Path(r["directory"]) / r["file"]).resolve() == source]
    check(len(matches) == 1, f"compile_commands has {len(matches)} entries for {source}")
    row = matches[0]
    cwd = Path(row["directory"]).resolve()
    argv = list(row["arguments"]) if "arguments" in row else shlex.split(row["command"])
    obj = staged / (source.name + ".o")
    original = (cwd / replace_flag(argv, "-o", obj)).resolve()
    replace_flag(argv, "-c", staged / name)
    # Prefer the staged op-plugin tree, then the quoted includes beside the original source.
    argv[1:1] = ["-I" + str(staged), "-iquote", str(source.parent)]
    if "-MF" in argv:
        replace_flag(argv, "-MF", obj.with_suffix(".d"))
    check(original.is_file(), f"missing baseline object {original}")
    return {"source": name, "cwd": str(cwd), "original_object": str(original), "object": str(obj), "argv": argv}


def relink_cmd(build, units, destination):
    commands = output(["ninja", "-t", "commands", TARGET], cwd=build)
    raw = commands.splitlines()[-1]
    check(raw.startswith(": && ") and raw.endswith(" && :"), "unexpected shell wrapper on Ninja link rule")
    argv = [s.replace(r"\$ORIGIN", "$ORIGIN") for s in shlex.split(raw[5:-5])]
    check(replace_flag(argv, "-o", destination) == TARGET, "unexpected Ninja link output")
    for unit in units:
        matches = [i for i, a in enumerate(argv) if a.endswith(".o") and
                   (build / a).resolve() == Path(unit["original_object"])]
        check(len(matches) == 1, f"expected direct final-link input for {unit['original_object']}, found {len(matches)}")
        argv[matches[0]] = unit["object"]
    return argv


def run_build(args):
    base = args.base.resolve(strict=True)
    checkout = base / "checkout"
    build = checkout / "build"
    plugin = checkout / "third_party/op-plugin"
    installer = args.installer.resolve(strict=True)
    out = args.output.resolve()
    check(not out.is_relative_to(base), "use an isolated output directory")
    out.mkdir(parents=True, exist_ok=True)
    package = out / "package"
    check(not package.exists(), f"package already exists: {package}")
    stage = Path(tempfile.mkdtemp(prefix="compile-", dir=out))
    receipt = {"base": str(base), "stage": str(stage), "installer": str(installer),
               "pta_sha": output(["git", "rev-parse", "HEAD"], cwd=checkout),
               "op_plugin_sha": output(["git", "rev-parse", "HEAD"], cwd=plugin),
               "installer_sha256": digest(installer),
               "input_files_sha256": {p.name: digest(p) for p in sorted(installer.parent.glob("index_scope.*"))},
               "original_sources_sha256": {}, "patched_sources_sha256": {},
               "status": "preparing"}
    receipt_file = out / "build-receipt.json"

    def save():
        receipt_file.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    try:
        check(receipt["pta_sha"] == PTA_SHA and receipt["op_plugin_sha"] == PLUGIN_SHA,
              "unexpected checkout or op-plugin revision")
        for name, expected in SOURCES.items():
            original, target = plugin / name, stage / name
            actual = digest(original)
            receipt["original_sources_sha256"][name] = actual
            check(actual == expected, f"base contents differ for {name}: {actual}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(original, target)
        subprocess.run([sys.executable, "-B", str(installer), str(stage)], check=True)
        receipt["patched_sources_sha256"] = {str(p.relative_to(stage)): digest(p) for p in sorted(stage.rglob("*"))
                                             if p.is_file()}
        commands_db = build / "compile_commands.json"
        receipt["compile_commands_sha256"] = digest(commands_db)
        db = json.loads(commands_db.read_text())
        units = [compile_unit(db, plugin, stage, name) for name in SOURCES]
        receipt["compile"] = units
        linked = stage / "libtorch_npu.so"
        command = relink_cmd(build, units, linked)
        receipt.update(link=command, link_cwd=str(build), status="compiling")
        save()

        def compile_one(unit):
            log = stage / (Path(unit["source"]).name + ".compile.log")
            with log.open("wb") as stream:
                result = subprocess.run(unit["argv"], cwd=unit["cwd"], stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                print(log.read_text(errors="replace")[-10000:], file=sys.stderr)
            result.check_returncode()

        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            list(pool.map(compile_one, units))
        for unit in units:
            unit["object_sha256"] = digest(Path(unit["object"]))
        receipt["status"] = "linking"
        save()
        subprocess.run(command, cwd=build, check=True)
        original_package = base / "package"
        relative_lib = Path("torch_npu/lib/libtorch_npu.so")
        original_lib = original_package / relative_lib
        receipt["original_package_library_sha256"] = digest(original_lib)
        staged_package = stage / "package"

        def skip_library(path, names):
            return {relative_lib.name} if Path(path) == original_lib.parent else set()

        shutil.copytree(original_package, staged_package, symlinks=True, ignore=skip_library)
        library = staged_package / relative_lib
        check(library.parent.resolve().is_relative_to(staged_package), "library destination outside package")
        os.replace(linked, library)
        receipt["library_sha256"] = digest(library)
        os.rename(staged_package, package)
        receipt.update(status="built", library=str(package / relative_lib), pythonpath=str(package))
        save()
        print(json.dumps({"PYTHONPATH": str(package), "sha256": receipt["library_sha256"], "receipt": str(receipt_file)}))
    except Exception as error:
        receipt.update(status="failed", error=str(error))
        save()
        raise


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--base", type=Path, default=BASE)
    cli.add_argument("--installer", type=Path,
                     default=Path(__file__).resolve().parent / "rebase/apply_rebase.py")
    cli.add_argument("--output", required=True, type=Path)
    cli.add_argument("--jobs", choices=(1, 2), type=int, default=2)
    run_build(cli.parse_args())
