"""Create a Python-identical TorchNPU overlay with one private native library."""
import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    ap.add_argument("library", help="candidate libtorch_npu.so inside training container")
    ap.add_argument("sha256")
    ap.add_argument("switch")
    ap.add_argument("--counters", action="store_true")
    ap.add_argument("--diagnostic-switch")
    args = ap.parse_args()
    study = Path(__file__).resolve().parent
    inner = "/artifacts/" + study.name
    overlay = study / "stock-overlay" / args.name
    if not overlay.exists():
        init_pid = subprocess.check_output(["docker", "inspect", "-f", "{{.State.Pid}}",
                                           "recsys-gr-ds019-npu8"], text=True).strip()
        root = Path("/proc") / init_pid / "root"
        stock, = (root / "usr/local").glob("python*/lib/python*/site-packages/torch_npu")
        shutil.copytree(stock, overlay / "torch_npu", symlinks=True)
        shutil.copyfile(root / args.library.lstrip("/"), overlay / "torch_npu/lib/libtorch_npu.so")
    key = "torch_npu/lib/libtorch_npu.so"
    with (overlay / key).open("rb") as f:
        assert hashlib.file_digest(f,"sha256").hexdigest() == args.sha256
    path = inner + "/stock-overlay/" + args.name
    for audit in (False, True):
        common = {"PYTHONPATH": path}
        libs = {path + "/" + key: args.sha256}
        if audit and args.counters:
            counter = inner + "/counter_dump.so"
            common.update(LD_PRELOAD=counter,INFRA_COUNTER_DIR="{output}")
            libs[counter] = hashlib.sha256((study / "counter_dump.so").read_bytes()).hexdigest()
        if args.diagnostic_switch:
            common[args.diagnostic_switch] = str(int(audit))
        recipe = {"common":common,"switch":args.switch,"A":"0","B":"1","libraries":libs}
        target = study / "recipes" / (args.name + ("-audit" if audit else "") + ".json")
        target.parent.mkdir(exist_ok=True)
        target.write_text(json.dumps(recipe, indent=2)+"\n")
if __name__ == "__main__":
    main()

