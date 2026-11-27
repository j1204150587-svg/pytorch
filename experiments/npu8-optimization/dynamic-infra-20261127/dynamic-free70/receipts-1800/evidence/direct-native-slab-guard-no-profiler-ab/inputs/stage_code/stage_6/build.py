"""Build the in-process side-slab host guard against this CPython interpreter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(out):
    out = out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cpp = Path(__file__).resolve().with_name("slab_guard.cpp")
    binary = out / ("_hstu_slab_guard" + sysconfig.get_config_var("EXT_SUFFIX"))
    receipt = {"python": sys.version, "source_sha256": digest(cpp), "library": binary.name}
    manifest = out / "build.json"
    if binary.is_file() and manifest.is_file():
        old = json.loads(manifest.read_text())
        if all(old.get(k) == v for k, v in receipt.items()) and digest(binary) == old.get("library_sha256"):
            print(binary)
            return
    include = sysconfig.get_paths()["include"]
    with tempfile.TemporaryDirectory(dir=out) as work:
        temp = Path(work) / binary.name
        if os.name == "nt":
            cmd = [os.getenv("CXX", "cl"), "/nologo", "/O2", "/std:c++17", "/MD", "/EHsc", "/LD",
                   "/I" + include, str(cpp), "/link", "/OUT:" + str(temp),
                   "/LIBPATH:" + str(Path(sys.base_prefix) / "libs")]
        else:
            cmd = [os.getenv("CXX", "c++"), "-O2", "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden",
                   "-I" + include, str(cpp), "-o", str(temp)]
        subprocess.run(cmd, cwd=work, check=True)
        temp.replace(binary)
    receipt.update(library_sha256=digest(binary), command=cmd)
    manifest.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(binary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    main(parser.parse_args().output)
