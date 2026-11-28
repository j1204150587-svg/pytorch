"""Reuse the pinned one-unit build and link recipe with a PTA source root."""
import argparse
import hashlib
from pathlib import Path
import subprocess

from install import SOURCE


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--builder", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    source = args.builder.resolve().read_text()
    anchor = "    plugin = checkout / \"third_party/op-plugin\""
    # The stock builder validates the plugin commit independently of the compile root.
    for old, new in [(anchor, anchor + "\n    source_root = checkout"),
                     ("original, target = plugin / name, stage / name",
                      "original, target = source_root / name, stage / name"),
                     ("compile_unit(db, plugin, stage, name)", "compile_unit(db, source_root, stage, name)"),
                     ("symlinks=True, ignore=skip_library)", "symlinks=True, ignore=skip_library, copy_function=os.link)")]:
        if source.count(old) != 1:
            raise ValueError("unrecognized pinned builder")
        source = source.replace(old, new)
    spec = {"__file__": str(args.builder), "__name__": "inline_view_native_builder"}
    exec(compile(source, str(args.builder), "exec"), spec)
    stock = subprocess.check_output(["git", "-C", str(spec["BASE"] / "checkout"), "show",
                                      spec["PTA_SHA"] + ":" + SOURCE])
    if hashlib.sha1(b"blob " + str(len(stock)).encode() + b"\0" + stock).hexdigest() != "c5d9fc2165ade45d6ce69151fe1acfd26e1c48eb":
        raise ValueError("unexpected PTA view source")
    spec["SOURCES"] = {SOURCE: hashlib.sha256(stock).hexdigest()}
    args.base = spec["BASE"]
    args.installer = Path(__file__).with_name("install.py")
    args.jobs = 1
    spec["run_build"](args)


if __name__ == "__main__":
    main()
