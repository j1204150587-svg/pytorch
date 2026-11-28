#!/usr/bin/env python3
"""Launch one pair through the archived runner, or act as its host-side docker shim."""

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time


HERE = Path(__file__).resolve().parent
REFERENCE = json.loads((HERE / "reference.json").read_text())
ARTIFACTS = Path("/data/recsys-gr-ds019-host88/artifacts")
SELECTOR = "RECSYS_DYNAMIC_INFRA_ONLY"
CONTAINER = "recsys-gr-ds019-npu8"
NATIVE_KEY = re.compile(r"(?:TORCH_NPU|NPU|ACLNN|ACL|HCCL|ASCEND)_[A-Z0-9_]+")


def check(condition, detail):
    if not condition:
        raise ValueError(detail)


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(directory, manifest, exact=False):
    if exact:
        entries = list(directory.rglob("*"))
        check(not any(p.is_symlink() for p in entries), f"symlink in workspace {directory}")
        names = {p.relative_to(directory).as_posix() for p in entries if p.is_file()}
        check(names == set(manifest), f"workspace file set differs: {names ^ set(manifest)}")
    for name, expected in manifest.items():
        path = directory / name
        check(path.is_file() and digest(path) == expected, f"source hash differs: {path}")


def native_library(root, absolute):
    check(absolute.startswith("/") and ".." not in Path(absolute).parts, f"invalid runtime path: {absolute}")
    return root / absolute.lstrip("/")


def prepare_recipe(recipe, root):
    check(set(recipe) == {"common", "switch", "A", "B", "libraries"},
          "recipe requires common, switch, A, B and libraries")
    common, key = recipe["common"], recipe["switch"]
    check(isinstance(common, dict) and NATIVE_KEY.fullmatch(key) and key not in common,
          "one separate native switch required")
    check(all(isinstance(recipe[v], str) for v in ("A", "B")), "A/B switch values must be strings")
    for name, value in common.items():
        check(isinstance(value, str) and (NATIVE_KEY.fullmatch(name) or name in
              {"LD_PRELOAD", "PYTHONPATH", "INFRA_COUNTER_DIR"}),
              f"unexpected common option {name}")
        if name == "INFRA_COUNTER_DIR":
            check(value == "{output}", "counter directory must be the current leg's {output}")
    libraries = recipe["libraries"]
    check(isinstance(libraries, dict) and libraries and any(p.endswith("/libtorch_npu.so") for p in libraries),
          "specify exact mapped libraries and SHA-256 including libtorch_npu.so")
    for name, expected in libraries.items():
        check(re.fullmatch("[0-9a-f]{64}", expected) and digest(native_library(root, name)) == expected,
              f"native library hash differs: {name}")
    if "LD_PRELOAD" in common:
        check(set(re.split(r"[: ]+", common["LD_PRELOAD"])) <= set(libraries), "untracked preload library")
    if "PYTHONPATH" in common:
        package_parent = native_library(root, common["PYTHONPATH"])
        check({p.name for p in package_parent.iterdir()} == {"torch_npu"}, "overlay must only contain torch_npu")
        stock = list((root / "usr/local").glob("python*/lib/python*/site-packages/torch_npu"))
        check(len(stock) == 1 and stock[0].is_dir(), "cannot locate one stock torch_npu package")
        def sources(pkg):
            # Only explicitly pinned native libraries may change. The overlay cannot add
            # a Python entry point, extension, sourceless bytecode or startup file.
            return {p.relative_to(pkg).as_posix(): digest(p) for p in pkg.rglob("*")
                    if p.is_file() and "__pycache__" not in p.relative_to(pkg).parts
                    and str(Path(common["PYTHONPATH"]) / "torch_npu" / p.relative_to(pkg)) not in libraries}
        overlay = package_parent / "torch_npu"
        for cached in overlay.rglob("*.pyc"):
            original = stock[0] / cached.relative_to(overlay)
            check(original.is_file() and digest(original) == digest(cached), f"unexpected cached bytecode: {cached}")
        check(sources(stock[0]) == sources(overlay), "overlay contents differ beyond explicitly pinned shared libraries")
        check(any(p.startswith(common["PYTHONPATH"].rstrip("/") + "/torch_npu/") for p in libraries),
              "no overlay library is pinned")


def mapped_process(init_pid, output, libraries, expected_env, stop, result):
    # Host-only observation. Inspect start-up once and stop polling before the steady window.
    container_group = Path(f"/proc/{init_pid}/cgroup").read_bytes()
    while not stop.is_set():
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            proc = Path("/proc") / name
            try:
                command = (proc / "cmdline").read_bytes().split(b"\0")
                if not any(Path(os.fsdecode(arg)).name == "train_deepspeed.py" for arg in command):
                    continue
                if os.fsencode(output) not in command or (proc / "cgroup").read_bytes() != container_group:
                    continue
                # torchrun also carries the training argv. Only attest the worker.
                env = dict(x.split(b"=", 1) for x in (proc / "environ").read_bytes().split(b"\0") if b"=" in x)
                if env.get(b"LOCAL_RANK") != b"0":
                    continue
                maps = (proc / "maps").read_text().splitlines()
                actual = {}
                overlay_paths = []
                overlay_root = any(row.split()[4] == "/" and " - overlay " in row
                                   for row in (proc / "mountinfo").read_text().splitlines())
                for path in libraries:
                    stat = native_library(proc / "root", path).stat()
                    for row in maps:
                        fields = row.split(maxsplit=5)
                        if len(fields) == 6 and int(fields[4]) == stat.st_ino:
                            major, minor = (int(x, 16) for x in fields[3].split(":"))
                            if os.makedev(major, minor) == stat.st_dev:
                                actual[path] = row
                                break
                            # Lower-layer ELF entries can report a backing device in
                            # maps while stat returns the container root overlay device.
                            if (overlay_root and fields[5] == path and path.startswith("/usr/local/")
                                    and digest(native_library(proc / "root", path)) == libraries[path]):
                                actual[path] = row
                                overlay_paths.append(path)
                                break
                if len(actual) != len(libraries):
                    continue
                observed = {key: os.fsdecode(env.get(os.fsencode(key), b"")) for key in expected_env}
                for key, expected in expected_env.items():
                    if key == "PYTHONPATH":
                        check(expected in observed[key].split(":"), f"CANN setup lost the PTA package path: {observed}")
                    else:
                        check(observed[key] == expected, f"worker environment differs: {observed}")
                namespace = next(line for line in (proc / "status").read_text().splitlines() if line.startswith("NSpid:"))
                result.update(pid=int(name), container_pid=int(namespace.split()[-1]), init_pid=init_pid,
                              libraries=actual, env=observed, overlay_lower_files=overlay_paths)
                return
            except (OSError, ProcessLookupError):
                continue
        stop.wait(.5)


def docker_shim():
    argv = sys.argv[1:]
    real_docker = os.environ["NPU8_HOST_DOCKER"]
    if argv[:3] != ["--host", "unix:///var/run/docker.sock", "exec"]:
        os.execv(real_docker, [real_docker, *argv])
    # The original runner's only docker exec is the actual training command.
    index = argv.index("./run_training.sh")
    check(argv[index - 1] == "bash" and argv[3] == "--workdir", "unexpected original docker exec")
    inner_work = Path(argv[4])
    check(inner_work.parts[:3] == ("/", "artifacts", "dynamic-shapes-20261127") and inner_work.name == "workspace",
          f"unexpected original dynamic workdir: {inner_work}")
    session = ARTIFACTS.joinpath(*inner_work.parts[2:-1])
    verify(session / "workspace", REFERENCE["workspace"], exact=True)
    container = read(session / "container.json")
    check(container["id"] == argv[index - 2], "container ID differs from original preflight")
    passed = argv[5:index - 2]
    check(all(passed[i] == "--env" for i in range(0, len(passed), 2)), "unexpected docker exec option")
    original = dict(word.split("=", 1) for word in passed[1::2])
    check(original.get(SELECTOR) in {"0", "1"} and original.get("TASK_QUEUE_ENABLE") == "1" and
          original.get("RECSYS_ZERO2_BATCH_OVERFLOW") == "1", "original training control mismatch")
    out_arg = argv[argv.index("--output-dir") + 1]
    leg = Path(out_arg).name
    variant = {"0": "A", "1": "B"}[original[SELECTOR]]
    check(leg in {"01-A", "01-B", "02-A", "02-B"} and leg[-1] == variant, "variant differs from output path")
    receipt_dir = Path(os.environ["NPU8_HOST_RECEIPTS"])
    folder = receipt_dir / leg
    folder.mkdir()
    recipe = read(receipt_dir / "recipe.json")
    root = Path(f"/proc/{container['init_pid']}/root")
    prepare_recipe(recipe, root)
    extra = {**recipe["common"], recipe["switch"]: recipe[variant]}
    if "INFRA_COUNTER_DIR" in extra:
        check(original.get("RECSYS_DYNAMIC_STATE_AUDIT") == "1", "counters may only run in the semantic audit")
        extra["INFRA_COUNTER_DIR"] = out_arg
    check(not (set(original) & set(extra)), "original environment would be overridden")
    effective = {k: v for k, v in original.items() if k != SELECTOR}
    effective.update(extra)
    actual = [real_docker, *argv[:5], *(arg for k,v in effective.items() for arg in ("--env", f"{k}={v}")),
              *argv[index - 2:]]
    write(folder / "dispatch.json", dict(original=argv, actual=actual, variant=variant, session=str(session),
                                         output=out_arg, workspace=REFERENCE["workspace"], libraries=recipe["libraries"]))
    stop = threading.Event()
    observed = {}
    monitor = threading.Thread(target=mapped_process, args=(container["init_pid"], out_arg, recipe["libraries"],
                                                        extra, stop, observed), daemon=True)
    monitor.start()
    try:
        code = subprocess.run(actual).returncode
    finally:
        stop.set()
        monitor.join()
        write(folder / "worker.json", observed)
    verify(session / "workspace", REFERENCE["workspace"], exact=True)
    for path, expected in recipe["libraries"].items():
        check(digest(native_library(root, path)) == expected, f"native library changed during run: {path}")
    check(observed, "did not attest requested native libraries and switches in the training worker")
    return code


def audit_results(entry):
    check(entry.get("audit_passed") and entry.get("audited_steps") == 8, "original numerical gate failed")
    run = Path(entry["directory"])
    steps = {}
    for folder in sorted(run.glob("0?-?")):
        rank = folder / "rank000"
        data, overflow = read(rank / "state_audit.json"), read(rank / "overflow_scan.json")
        metrics = read(rank / "metrics.json")["steps"]
        check(len(metrics) == 8 and len(data["gradients_and_norms"]) == 8, "missing original audit steps")
        check(overflow["valid"] and overflow["fast_calls"] == 8 and not overflow["fallback_calls"]
              and not overflow["errors"], "01 did not activate for exactly eight steps")
        check(data["adamw_steps"] and all(x == 8 for x in data["adamw_steps"]), "expected eight finite weight updates")
        steps[folder.name[-1]] = [[s["step"], s["batch_size"], s["seq_len"]] for s in metrics]
    check(steps["A"] == steps["B"], "audit shape order differs")
    return {"steps_and_shapes": steps["A"], "finite_updates": 8, "observed_overflow_skips": 0,
            "induced_nonfinite": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="fresh host receipts directory")
    parser.add_argument("--mode", choices=("audit", "no-profiler", "l0", "l1"), required=True)
    parser.add_argument("--order", choices=("ab", "ba"), default="ab")
    args = parser.parse_args()
    # Serialize study drivers before the archived runner starts its bounded card-lock wait.
    # The archived runner still owns the shared NPU reservation for the entire pair.
    admission = Path("/run/lock/recsys-gr-ds019-npu8-infra-study.lock").open("a")
    fcntl.flock(admission, fcntl.LOCK_EX)
    check(args.mode != "audit" or args.order == "ab", "the original audit reader expects AB")
    runner = args.runner.resolve()
    verify(runner, REFERENCE["runner"])
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.recipe, args.out / "recipe.json")
    real = shutil.which("docker")
    check(real, "docker not found on host")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location("original_suite", runner / "run_suite.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory(prefix="npu8-docker-") as shim:
        (Path(shim) / "docker").symlink_to(Path(__file__).resolve())
        os.environ.update(NPU8_HOST_DOCKER=os.path.realpath(real), NPU8_HOST_RECEIPTS=str(args.out),
                          PATH=shim + os.pathsep + os.environ["PATH"])
        entry = module.run_pair("INFRA_ONLY", "no-profiler" if args.mode == "audit" else args.mode,
                                args.order, audit=args.mode == "audit")
    write(args.out / "result.json", entry)
    check(not entry["exit"], f"original runner exit {entry['exit']}")
    if args.mode == "audit":
        entry["semantic_checks"] = audit_results(entry)
        recipe = read(args.out / "recipe.json")
        if "INFRA_COUNTER_DIR" in recipe["common"]:
            counters = {}
            for dispatch in args.out.glob("??-?/dispatch.json"):
                leg = dispatch.parent.name
                worker_pid = read(dispatch.parent / "worker.json")["container_pid"]
                counters[leg[-1]] = {p.name.rsplit("-", 1)[0]: read(p) for p in
                                     (Path(entry["directory"]) / leg).glob(f"*-{worker_pid}.json")}
                check(counters[leg[-1]], f"missing native diagnostic for the actual training PID on {leg}")
            entry["native_counters"] = counters
    check(len(list(args.out.glob("??-?/worker.json"))) == 2, "missing effective dispatch receipts")
    verify(runner, REFERENCE["runner"])
    write(args.out / "result.json", entry)
    print(json.dumps(entry), flush=True)
    admission.close()
    return 0


if __name__ == "__main__":
    sys.exit(docker_shim() if Path(sys.argv[0]).name == "docker" else main())
