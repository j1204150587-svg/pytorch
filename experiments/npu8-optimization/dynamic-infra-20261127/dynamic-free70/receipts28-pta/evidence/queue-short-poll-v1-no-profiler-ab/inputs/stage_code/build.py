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
                     ("compile_unit(db, plugin, stage, name)", "compile_unit(db, source_root, stage, name)")]:
        if source.count(old) != 1:
            raise ValueError("unrecognized pinned builder")
        source = source.replace(old, new)
    spec = {"__file__": str(args.builder), "__name__": "short_poll_native_builder"}
    exec(compile(source, str(args.builder), "exec"), spec)
    stock = subprocess.check_output(["git", "-C", str(spec["BASE"] / "checkout"), "show",
                                      spec["PTA_SHA"] + ":" + SOURCE])
    if hashlib.sha1(b"blob " + str(len(stock)).encode() + b"\0" + stock).hexdigest() != "5afef4e0ab9ad0c2b5af0fdc56561a5df0c06435":
        raise ValueError("unexpected PTA queue source")
    spec["SOURCES"] = {SOURCE: hashlib.sha256(stock).hexdigest()}
    args.base = spec["BASE"]
    args.installer = Path(__file__).with_name("install.py")
    args.jobs = 1
    spec["run_build"](args)


if __name__ == "__main__":
    main()
