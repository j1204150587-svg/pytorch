"""Create the one-library overlay, gated counter reader, and shared-control recipes."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
LIB = Path("torch_npu/lib/libtorch_npu.so")
SRC_HASH = "4e4adcd8348f437abefb6f48c14b224f4b1b5342b145ddd4b745964f45ec2d8c"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", default="copy-rebase")
    p.add_argument("--build", default="build")
    p.add_argument("--diagnostic", action="store_true")
    p.add_argument("--addresses", action="store_true")
    p.add_argument("--consumer", action="store_true")
    args = p.parse_args()
    artifact = "/artifacts/infra-only-20261128"
    before = STUDY / "stock-overlay/embedding"
    after = STUDY / "stock-overlay" / args.name
    assert digest(before / LIB) == SRC_HASH
    linked = HERE / args.build / "libtorch_npu.so"
    build = json.loads((HERE / args.build / "build-receipt.json").read_bytes())
    checksum = digest(linked)
    assert build["library_sha256"] == checksum and build["parent_sha256"] == SRC_HASH and build["status"] == "built"
    if not after.exists():
        with tempfile.TemporaryDirectory(dir=after.parent) as work:
            pending = Path(work) / "candidate"
            def skip_native(path, names):
                return {LIB.name} if Path(path) == (before / LIB).parent else set()
            shutil.copytree(before, pending, symlinks=True, copy_function=os.link, ignore=skip_native)
            shutil.copyfile(linked, pending / LIB)
            pending.rename(after)
    assert digest(after / LIB) == checksum
    original = (STUDY / "counter_dump.c").read_text()
    target = 'const char *symbols[] = {"npu_addmm_rebase_counter", "npu_mm_rebase_counter"};'
    assert original.count(target) == 1
    modified = original.replace(target, 'const char *symbols[] = {"npu_inplace_copy_rebase_counter", '
                                      '"npu_embedding_backward_rebase_counter", "npu_embedding_forward_rebase_counter"};')
    (HERE / "counter_copy.c").write_text(modified)
    inner = artifact + "/copy_rebase"
    run = ["docker", "run", "--rm", "--runtime=runc", "--network=none", "--cpus", "1",
           "-v", f"{HERE}:{inner}", "recsys-gr-deepspeed:host88-ds0.19.0-torch2.10.0",
           "gcc", "-O2", "-shared", "-fPIC", inner + "/counter_copy.c", "-o", inner + "/counter_copy.so", "-ldl"]
    if not (HERE / "counter_copy.so").exists():
        subprocess.run(run, check=True)
    for audit in ((True,) if args.diagnostic or args.addresses else (False, True)):
        prefix = artifact + "/stock-overlay/" + args.name
        common = {"PYTHONPATH":prefix, "ACLNN_CACHE_LIMIT":"65536", "NPU_DYNAMIC_REBASE_INDICES":"1",
                  "NPU_DYNAMIC_REBASE_FORWARD":"1", "NPU_DYNAMIC_REBASE_COUNT":str(int(audit))}
        libraries = {prefix + "/" + str(LIB):checksum}
        if args.diagnostic:
            common["NPU_INPLACE_COPY_DIAG"] = "1"
        if args.addresses:
            common["NPU_INPLACE_COPY_REBASE"] = "0"
        if audit:
            counter = inner + "/counter_copy.so"
            common.update(LD_PRELOAD=counter, INFRA_COUNTER_DIR="{output}")
            libraries[counter] = digest(HERE / "counter_copy.so")
        switch = "NPU_INPLACE_COPY_ADDRESS" if args.addresses else "NPU_INPLACE_COPY_REBASE"
        if args.consumer:
            switch = "NPU_INPLACE_COPY_CONSUMER"
        config = dict(common=common, switch=switch, A="0", B="1", libraries=libraries)
        name = args.name + ("-audit.json" if audit else ".json")
        (STUDY / "recipes" / name).write_text(json.dumps(config, indent=2) + "\n")


if __name__ == "__main__":
    main()
