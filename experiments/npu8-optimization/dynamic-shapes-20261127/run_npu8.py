#!/usr/bin/env python3
"""Stage or run the host88 logical NPU 8 study. The check command is offline."""

import argparse
from datetime import datetime, timezone
import gzip
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import tempfile

sys.dont_write_bytecode = True
from evidence import (files_sha256, idle_card, read_json, require, sha256, validate_overflow_scan,
                      validate_step_csv, validate_training, write_json, write_sha256)


HERE = Path(__file__).resolve().parent
ROOT = Path("/data/recsys-gr-ds019-host88")
CONTAINER = "recsys-gr-ds019-npu8"
IMAGE = "recsys-gr-deepspeed:host88-ds0.19.0-torch2.10.0"
DOCKER = ["docker", "--host", "unix:///var/run/docker.sock"]
LOCK = Path("/run/lock/recsys-gr-ds019-npu8.lock")
DEVICE = Path("/dev/davinci8")
STUDY = "dynamic-shapes-20261127"
FILES = ("train_deepspeed.py", "model.py", "run_training.sh", "env.sh",
         "configs/model_many_tables.json", "configs/deepspeed_zero2_bf16_overflow_on.json")
OVERFLOW_SWITCH = "RECSYS_ZERO2_BATCH_OVERFLOW"
ORDERS = {"baseline": ("A",), "aa": ("A", "A"), "ab": ("A", "B"), "ba": ("B", "A"), "screen": ("B",)}
STOP = None


class RunError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_exit(path, status):
    path.write_text(f"{status}\n", encoding="ascii")


def capture(argv, cwd=None, timeout=60):
    return subprocess.run(argv, cwd=cwd, capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, check=False)


def save_capture(path, result):
    path.write_text(result.stdout + result.stderr, encoding="utf-8")
    write_exit(path.with_suffix(".exit"), result.returncode)


def snapshot(source, dest):
    dest.mkdir()
    for name in FILES:
        original, copy = source / name, dest / name
        require(original.is_file() and not original.is_symlink()
                and original.resolve().is_relative_to(source.resolve()), f"unsafe or missing source: {original}")
        copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copy)


def apply(workspace, patch, logfile):
    argv = ["git", "apply", "--no-index", "--whitespace=nowarn"]
    for extra in (["--check"], []):
        result = capture([*argv, *extra, str(patch)], cwd=workspace)
        with logfile.open("a", encoding="utf-8") as log:
            log.write(f"{shlex.join([*argv, *extra, str(patch)])} -> {result.returncode}\n")
            log.write(result.stdout + result.stderr)
        require(result.returncode == 0, f"patch failed: {patch.name}; see {logfile}")


def stash(path, dest, digest):
    shutil.copyfile(path, dest)
    require(digest is None or sha256(dest) == digest, f"SHA-256 mismatch: {path}")
    return dest


def prepare(source, session, args):
    inputs = session / "inputs"
    inputs.mkdir()
    workspace = session / "workspace"
    snapshot(source, workspace)
    if getattr(args, "shape_schedule", None):
        stash(args.shape_schedule, workspace / "configs/dynamic_shapes_npu8.json", args.shape_schedule_sha256)
    model = read_json(workspace / FILES[4])
    ds = read_json(workspace / FILES[5])
    require(tuple(model[key] for key in ("num_tables", "table_rows", "features_per_table", "seq_len", "embedding_dim", "num_layers"))
            == (1024, 64, 4, 32, 64, 4), "unexpected source model configuration")
    require(ds["zero_optimization"]["stage"] == 2 and not ds["zero_optimization"].get("offload_optimizer")
            and ds["gradient_accumulation_steps"] == 1 and ds["gradient_clipping"] == 1.0
            and ds["bf16"]["enabled"] is True and ds["bf16"]["check_grad_overflow"] is True,
            "unexpected source DeepSpeed configuration")
    original = files_sha256(workspace)
    write_sha256(inputs / "original.sha256", original)
    logfile = session / "patches.log"
    if args.candidate_patch:
        candidate = stash(args.candidate_patch, inputs / "candidate.patch", args.candidate_patch_sha256)
        module = stash(args.candidate_module, inputs / "overflow.py", args.candidate_module_sha256)
        write_sha256(inputs / "candidate-files.sha256", {path.name: sha256(path) for path in (candidate, module)})
        apply(workspace, candidate, logfile)
        patched = files_sha256(workspace)
        changed = {name for name in original.keys() | patched.keys() if original.get(name) != patched.get(name)}
        require(changed == {"train_deepspeed.py"} and "train_deepspeed.py" in patched,
                f"candidate patch must only change the training hook: {sorted(changed)}")
        shutil.copyfile(module, workspace / module.name)
    steady = stash(HERE / "steady_window.patch", inputs / "steady_window.patch", None)
    apply(workspace, steady, logfile)
    if getattr(args, "extension_patch", None):
        extension = stash(args.extension_patch, inputs / "extension.patch", args.extension_patch_sha256)
        before = files_sha256(workspace)
        apply(workspace, extension, logfile)
        after = files_sha256(workspace)
        changed = {name for name in before.keys() | after.keys() if before.get(name) != after.get(name)}
        require(changed and changed <= {"train_deepspeed.py", "model.py"} and before.keys() == after.keys(),
                f"extension may only change the training hook or model: {sorted(changed)}")
        name = args.extension_module.name
        require(re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*\.py", name) is not None and not (workspace / name).exists(),
                f"invalid extension module name: {name}")
        module = stash(args.extension_module, inputs / name, args.extension_module_sha256)
        shutil.copyfile(module, workspace / name)
        if args.shared_norm_module:
            shared = stash(args.shared_norm_module, inputs / "zero2_batch_norm.py", args.shared_norm_sha256)
            shutil.copyfile(shared, workspace / shared.name)
        if args.pta_wrapper_module:
            wrapper = stash(args.pta_wrapper_module, inputs / "pta_wrapper.py", args.pta_wrapper_sha256)
            shutil.copyfile(wrapper, workspace / wrapper.name)
            graph_function = stash(args.pta_graph_function, inputs / "pta_graph_function.py", args.pta_graph_sha256)
            shutil.copyfile(graph_function, workspace / graph_function.name)
            if args.local_step:
                local_step = stash(args.local_step_module, inputs / "local_step.py", args.local_step_sha256)
                shutil.copyfile(local_step, workspace / local_step.name)
    if args.prefetch_patch:
        prefetch_patch = stash(args.prefetch_patch, inputs / "input-prefetch.patch", args.prefetch_patch_sha256)
        prefetch_module = stash(args.prefetch_module, inputs / "input_prefetch.py", args.prefetch_module_sha256)
        before = files_sha256(workspace)
        apply(workspace, prefetch_patch, logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"train_deepspeed.py"}, "input prefetch patch must only change the training entry")
        shutil.copyfile(prefetch_module, workspace / prefetch_module.name)
    if args.sidepack_dir:
        source = args.sidepack_dir
        manifest = stash(source / "sidepack-files.sha256", inputs / "sidepack-files.sha256", args.sidepack_sha256)
        hashes = {name: digest for digest, name in (line.split("  ", 1) for line in manifest.read_text().splitlines())}
        require(hashes.keys() == {"graph-sidepack.patch", "graph_sidepack.py", "sidepack_npu.py"}
                and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values()),
                "unexpected graph sidepack inputs")
        for name, digest in hashes.items():
            stash(source / name, inputs / name, digest)
        before = files_sha256(workspace)
        apply(workspace, inputs / "graph-sidepack.patch", logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"model.py", "graphed_hstu.py"}, "graph sidepack patch must change only the model and replay hook")
        for name in ("graph_sidepack.py", "sidepack_npu.py"):
            shutil.copyfile(inputs / name, workspace / name)
    if args.direct_input_patch:
        direct_patch = stash(args.direct_input_patch, inputs / "direct-input.patch", args.direct_input_patch_sha256)
        direct_module = stash(args.direct_input_module, inputs / "direct_graph_input.py", args.direct_input_module_sha256)
        before = files_sha256(workspace)
        apply(workspace, direct_patch, logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"train_deepspeed.py", "graphed_hstu.py"}, "direct H2D patch must only change the training entry and replay hook")
        shutil.copyfile(direct_module, workspace / direct_module.name)
    if args.direct_grad_dir:
        source = args.direct_grad_dir
        manifest = stash(source / "direct-grad-files.sha256", inputs / "direct-grad-files.sha256", args.direct_grad_sha256)
        hashes = {name: digest for digest, name in (line.split("  ", 1) for line in manifest.read_text().splitlines())}
        require(hashes.keys() == {"direct-grad.patch", "direct_gradient.py"}
                and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values()),
                "unexpected direct gradient inputs")
        for name, digest in hashes.items():
            stash(source / name, inputs / name, digest)
        before = files_sha256(workspace)
        apply(workspace, inputs / "direct-grad.patch", logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"graphed_hstu.py"}, "direct gradient patch must change only the replay and partition hook")
        shutil.copyfile(inputs / "direct_gradient.py", workspace / "direct_gradient.py")
    if args.bound_stats_dir:
        source = args.bound_stats_dir
        manifest = stash(source / "bound-stats-files.sha256", inputs / "bound-stats-files.sha256", args.bound_stats_sha256)
        hashes = {name: digest for digest, name in (line.split("  ", 1) for line in manifest.read_text().splitlines())}
        require(hashes.keys() == {"pta-bound-stats.patch", "pta_bound_stats.py"}
                and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values()),
                "unexpected bound statistics inputs")
        for name, digest in hashes.items():
            stash(source / name, inputs / name, digest)
        before = files_sha256(workspace)
        apply(workspace, inputs / "pta-bound-stats.patch", logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"pta_wrapper.py"}, "bound statistics patch must only change the PTA wrapper")
        shutil.copyfile(inputs / "pta_bound_stats.py", workspace / "pta_bound_stats.py")
    if args.pinned_input_dir:
        source = args.pinned_input_dir
        manifest = stash(source / "pinned-input-files.sha256", inputs / "pinned-input-files.sha256", args.pinned_input_sha256)
        hashes = {name: digest for digest, name in (line.split("  ", 1) for line in manifest.read_text().splitlines())}
        require(hashes.keys() == {"pinned-input.patch", "pinned_input.py"}
                and all(re.fullmatch(r"[0-9a-f]{64}", digest) for digest in hashes.values()),
                "unexpected pinned input inputs")
        for name, digest in hashes.items():
            stash(source / name, inputs / name, digest)
        before = files_sha256(workspace)
        apply(workspace, inputs / "pinned-input.patch", logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"train_deepspeed.py"}, "pinned input patch must only change the training entry")
        shutil.copyfile(inputs / "pinned_input.py", workspace / "pinned_input.py")
    if args.local_stats_patch:
        local_stats = stash(args.local_stats_patch, inputs / "pta-local-stats.patch", args.local_stats_sha256)
        before = files_sha256(workspace)
        apply(workspace, local_stats, logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"pta_wrapper.py"}, "local statistics patch must only change the PTA wrapper")
    if args.graph_loss_patch:
        graph_loss = stash(args.graph_loss_patch, inputs / "graph-loss.patch", args.graph_loss_sha256)
        before = files_sha256(workspace)
        apply(workspace, graph_loss, logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"graphed_hstu.py", "train_deepspeed.py"}, "loss capture patch must change only the replay and training entry")
    if args.grouped_ops_patch:
        grouped_ops = stash(args.grouped_ops_patch, inputs / "grouped-small-ops.patch", args.grouped_ops_sha256)
        before = files_sha256(workspace)
        apply(workspace, grouped_ops, logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"model.py"}, "small op reordering patch must change only the model")
    if args.joint_stats_patch:
        joint_stats = stash(args.joint_stats_patch, inputs / "pta-joint-stats.patch", args.joint_stats_sha256)
        before = files_sha256(workspace)
        apply(workspace, joint_stats, logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"pta_wrapper.py"}, "joint statistics patch must change only the PTA wrapper")
    if args.overlap_input_patch:
        overlap_patch = stash(args.overlap_input_patch, inputs / "overlap-input.patch", args.overlap_input_sha256)
        overlap_module = stash(args.overlap_input_module, inputs / "overlap_input.py", args.overlap_module_sha256)
        before = files_sha256(workspace)
        apply(workspace, overlap_patch, logfile)
        after = files_sha256(workspace)
        require(before.keys() == after.keys() and {name for name in before if before[name] != after[name]}
                == {"train_deepspeed.py"}, "input overlap patch must change only the training entry")
        shutil.copyfile(overlap_module, workspace / overlap_module.name)
    if args.graph_guards_patch:
        guards = stash(args.graph_guards_patch, inputs / "graph-guards-bundle.patch", args.graph_guards_sha256)
        before = files_sha256(workspace)
        apply(workspace, guards, logfile)
        after = files_sha256(workspace)
        require(after.keys() - before.keys() == {"graph_guards.py"} and not before.keys() - after.keys()
                and {name for name in before if before[name] != after[name]} == {"graphed_hstu.py"},
                "graph guards patch must change only the replay and guard module")
    for source_file in workspace.glob("*.py"):
        compile(source_file.read_bytes(), str(source_file), "exec")
    work_hash = files_sha256(workspace)
    write_sha256(session / "workspace.sha256", work_hash)
    return workspace, work_hash


def inspect_container(artifacts, session):
    result = capture([*DOCKER, "inspect", "--type", "container", CONTAINER])
    require(result.returncode == 0, result.stderr.strip())
    docker = json.loads(result.stdout)[0]
    host, config, state = docker["HostConfig"], docker["Config"], docker["State"]
    devices = [item for item in host["Devices"] if re.fullmatch(r"/dev/davinci\d+", item["PathOnHost"])
               or re.fullmatch(r"/dev/davinci\d+", item["PathInContainer"])]
    evidence = {"id": docker["Id"], "image_ref": config["Image"], "image_id": docker["Image"],
                "running": state["Running"], "init_pid": state["Pid"], "init_args": config["Cmd"],
                "devices": host["Devices"], "privileged": host["Privileged"], "device_requests": host.get("DeviceRequests"),
                "mounts": docker["Mounts"], "recsys_npu_ids": [item for item in config["Env"] if item.startswith("RECSYS_NPU_IDS=")],
                "cpu_quota": host["NanoCpus"], "memory_bytes": host["Memory"], "cpuset": host["CpusetCpus"]}
    write_json(session / "container.json", evidence)
    require(state["Running"] and config["Image"] == IMAGE and not host["Privileged"]
            and not host.get("DeviceRequests") and config["Cmd"] == ["sleep", "infinity"],
            "unexpected container state, image, permissions or init command")
    require([(item["PathOnHost"], item["PathInContainer"]) for item in devices] == [(str(DEVICE), str(DEVICE))]
            and evidence["recsys_npu_ids"] == ["RECSYS_NPU_IDS=8"], "unexpected container NPU mapping")
    mounts = [mount for mount in docker["Mounts"] if mount["Destination"] == "/artifacts"]
    require(len(mounts) == 1 and mounts[0]["Type"] == "bind" and mounts[0]["RW"]
            and Path(mounts[0]["Source"]).resolve() == artifacts, "unexpected /artifacts bind mount")
    return docker["Id"], state["Pid"]


def check_card(out, marker, container, init_pid, orphan_pid=None):
    checked = utc()
    mapping = capture(["npu-smi", "info", "-m"], timeout=20)
    save_capture(out / f"{marker}.npu-smi-mapping.log", mapping)
    smi = capture(["npu-smi", "info"], timeout=20)
    save_capture(out / f"{marker}.npu-smi.log", smi)
    holders = capture(["fuser", str(DEVICE)], timeout=20)
    save_capture(out / f"{marker}.fuser.log", holders)
    top = capture([*DOCKER, "top", container, "-eo", "pid,ppid,comm"], timeout=20)
    save_capture(out / f"{marker}.docker-top.log", top)
    require(mapping.returncode == smi.returncode == top.returncode == 0
            and not (mapping.stderr + smi.stderr).strip(), "npu-smi mapping, status or docker top failed")
    require(holders.returncode == 1 and not (holders.stdout + holders.stderr).strip(),
            "device open by another process or fuser could not confirm idle")
    pids = [int(row.split()[0]) for row in top.stdout.splitlines()[1:] if row.strip()]
    require(pids == [init_pid], f"training container has other processes: {pids}")
    card = idle_card(smi.stdout, mapping.stdout, 8, orphan_pid)
    if card["orphan_context"]:
        require(not Path(f"/proc/{orphan_pid}").exists(), "diagnostic orphan PID is live on the host")
        utilization = capture(["npu-smi", "info", "-t", "usages", "-i", str(card["board"]),
                               "-c", str(card["chip"])], timeout=20)
        save_capture(out / f"{marker}.npu-smi-usages.log", utilization)
        require(utilization.returncode == 0 and all(re.search(
            rf"^\s*{unit} Usage Rate\(%\)\s*:\s*0\s*$", utilization.stdout, re.M)
            for unit in ("Aicore", "Aivector", "Aicube", "Aicpu")), "diagnostic orphan context has active compute")
    write_json(out / f"{marker}.card.json", {"utc": checked, **card, "docker_top_pids": pids})
    return card["orphan_context"]


def training_options(args, inner_out):
    mode = args.mode
    profile = mode != "no-profiler"
    flags = {"model_config": FILES[4], "deepspeed_config": FILES[5], "steps": 8 if profile else args.warmup + args.measure,
             "micro_batch_size": 2, "seed": 1234, "learning_rate": 0.001, "weight_decay": 0.01,
             "log_interval": 1 if args.overlap_audit else (4 if profile else 10), "output_dir": inner_out, "profile": profile,
             "profile_active": {"l0": 3, "l1": 2, "no-profiler": 5}[mode], "profile_level1": mode == "l1",
             "profile_stack": False, "save_checkpoint": False, "dry_run": False,
             "steady_warmup_steps": 0 if profile else args.warmup, "steady_measure_steps": 0 if profile else args.measure}
    if args.shape_schedule:
        flags["shape_schedule"] = "configs/dynamic_shapes_npu8.json"
    if args.audit_state:
        flags["log_interval"] = 1
    cli = []
    for key, value in flags.items():
        option = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cli.append(option)
        else:
            cli.extend((option, str(value)))
    return flags, cli


def use_direct_gradient(args, variant):
    return variant == "B" or any((args.prefetch_patch, args.bound_stats_dir, args.local_stats_patch,
                                  args.cpu_affinity, args.graph_loss_patch, args.overlap_input_patch))


def analyze(out, workspace, args, flags, variant):
    result = validate_training(out, workspace, args.mode, flags)
    if args.shape_schedule:
        rank = out / "rank000"
        norm = (rank / "norm_stats.json").is_file()
        partition = (rank / "fast_partition.json").is_file()
        require(norm == (variant == "B" and args.extension_switch in ("RECSYS_DYNAMIC_NORM", "RECSYS_DYNAMIC_BOTH")),
                "dynamic norm activation mismatch")
        require(partition == (variant == "B" and args.extension_switch in ("RECSYS_DYNAMIC_PARTITION", "RECSYS_DYNAMIC_BOTH")),
                "dynamic partition activation mismatch")
        require((rank / "state_audit.json").is_file() == args.audit_state and not (rank / "hstu_graph.json").exists(),
                "unexpected graph replay or state audit")
        result["dynamic_partition"] = partition
        result["dynamic_norm"] = norm
    if args.grouped_ops_patch:
        model = read_json(out / "rank000" / "model_summary.json")
        require(model["grouped_small_ops"] is (variant == "B"), "small op reordering activation disagrees with experiment")
    if args.overlap_input_patch:
        summary = read_json(out / "rank000" / "metrics.json")["summary"]
        active = variant == "B" or bool(args.graph_guards_patch)
        uses = flags["steps"] - 1 - int(bool(flags["steady_measure_steps"])) if active else 0
        require(summary["overlap_input"] is active and summary["overlap_input_hits"] == uses,
                "input overlap activation disagrees with experiment")
        audit = out / "rank000" / "overlap_input_audit.json"
        require(audit.is_file() is args.overlap_audit, "input audit activation disagrees with experiment")
        if args.overlap_audit:
            data = read_json(audit)
            require(len(data["input_sha256"]) == flags["steps"] and data["input_sha256"] == data["device_input_sha256"],
                    "input hashes did not survive the device transfer")
    overflow_variant = "B" if args.queue_strategy or args.extension_patch else variant
    overflow_scan = validate_overflow_scan(out / "rank000" / "overflow_scan.json", overflow_variant, flags["steps"])
    if overflow_scan is not None:
        result["overflow_scan"] = overflow_scan
    if args.pta_wrapper_module:
        path = out / "rank000" / "pta_wrapper.json"
        active = bool(args.local_step) or variant == "B" or args.pta_control == "eager"
        require(path.is_file() == active, "PTA wrapper activation disagrees with experiment")
        if active:
            evidence = read_json(path)
            require(evidence["scope"] == args.pta_scope and evidence["graph"] is (bool(args.local_step) or variant == "B")
                    and all(evidence[phase] == (flags["steps"] if args.pta_scope in (phase, "both") else 0)
                            for phase in ("norm", "overflow")), "incomplete PTA wrapper calls")
            result["pta_wrapper"] = evidence
            if args.bound_stats_dir:
                require(evidence["bound_stats"] is (variant == "B"), "bound PTA statistics activation disagrees with experiment")
            if args.local_stats_patch:
                require(evidence["rank_one_stats"] is (variant == "B" or bool(
                    args.grouped_ops_patch or args.joint_stats_patch or args.overlap_prefetch)),
                        "local statistics activation disagrees with experiment")
            if args.joint_stats_patch:
                joint = variant == "B"
                require(evidence["joint_stats"] is joint and all(
                    evidence[name] == (flags["steps"] if joint else 0) for name in ("joint_replays", "joint_norm_used"))
                    and evidence["joint_norm_discarded"] == 0, "incomplete joint statistics replay")
        local_step_path = out / "rank000" / "zero2_local_step.json"
        local_step_active = bool(args.local_step) and (variant == "B" or bool(args.prefetch_patch or args.sidepack_dir))
        require(local_step_path.is_file() == local_step_active, "local optimizer activation disagrees with experiment")
        if local_step_active:
            evidence = read_json(local_step_path)
            require(evidence["mode"] == args.local_step and all(
                evidence[phase] == (flags["steps"] if args.local_step in (phase, "both") else 0)
                for phase in ("flatten", "aliases")), "incomplete local optimizer calls")
            result["zero2_local_step"] = evidence
    if args.prefetch_patch:
        summary = read_json(out / "rank000" / "metrics.json")["summary"]
        require(summary["cpu_input_prefetch"] is use_cpu_input_prefetch(args, variant),
                "input prefetch activation disagrees with experiment")
        result["cpu_input_prefetch"] = summary["cpu_input_prefetch"]
        if args.direct_input_patch:
            direct = variant == "B"
            require(summary["direct_h2d"] is direct and summary["direct_h2d_hits"] == (flags["steps"] if direct else 0),
                    "direct H2D activation disagrees with experiment")
            result["direct_h2d_hits"] = summary["direct_h2d_hits"]
        if args.pinned_input_dir:
            pinned = variant == "B"
            require(summary["pinned_input"] is pinned and summary["pinned_input_hits"] == (flags["steps"] if pinned else 0),
                    "pinned input activation disagrees with experiment")
            result["pinned_input_hits"] = summary["pinned_input_hits"]
    if args.sidepack_dir:
        graph = read_json(out / "rank000" / "hstu_graph.json")
        require(graph["sidepack"] is (variant == "B" or bool(args.direct_grad_dir))
                and graph["counts"] == {"forward": flags["steps"], "backward": flags["steps"]},
                "graph sidepack activation disagrees with experiment")
        result["graph_sidepack"] = graph["sidepack"]
        if args.graph_loss_patch:
            require(graph["graph_loss"] is (variant == "B"), "graph loss activation disagrees with experiment")
        if args.graph_guards_patch:
            require(graph["fast_guards"] is (variant == "B"), "fast graph guards activation disagrees with experiment")
        if args.direct_grad_dir:
            direct = use_direct_gradient(args, variant)
            require(graph["direct_gradient"] is direct
                    and graph["direct_gradient_hits"] == (flags["steps"] if direct else 0),
                    "direct gradient activation disagrees with experiment")
            result["direct_gradient_hits"] = graph["direct_gradient_hits"]
    if args.mode != "no-profiler":
        traces = sorted((out / "rank000" / "profile").rglob("trace_view.json"))
        require(len(traces) == 1 and traces[0].stat().st_size > 0, "expected one nonempty raw trace")
        trace = traces[0]
        csv_path = out / "step_trace_time.csv"
        shutil.copyfile(trace.with_name(csv_path.name), csv_path)
        result["profile_csv"] = validate_step_csv(csv_path, flags["profile_active"])
        with trace.open("rb") as source, gzip.open(out / "trace_view.json.gz", "wb", compresslevel=2) as dest:
            shutil.copyfileobj(source, dest)
        analyzer = out.parent / "inputs" / "analyze_trace.py"
        with (out / "trace_analysis.log").open("wb") as log:
            completed = subprocess.run([sys.executable, str(analyzer), "trace_view.json.gz", "-o", "trace_analysis.json"],
                                       cwd=out, stdout=log, stderr=subprocess.STDOUT, check=False, start_new_session=True)
        write_exit(out / "trace_analysis.exit", completed.returncode)
        require(completed.returncode == 0, "trace analyzer failed")
        report = read_json(out / "trace_analysis.json")
        expected = [f"ProfilerStep#{step}" for step in result["profile_csv"]["steps"]]
        require([row["name"] for row in report["steps"]] == expected and report["compute_task_count"] > 0,
                "trace analysis has incomplete step markers or no device compute")
    write_json(out / "validated.json", result)
    return result


def defer_stop(signum, _frame):
    global STOP
    STOP = signum


def stop_if_requested():
    if STOP is not None:
        raise RunError(128 + STOP, f"stop requested by signal {STOP}")


def use_cpu_input_prefetch(args, variant):
    if args.overlap_prefetch:
        return variant == "A"
    return variant == "B" or any((args.direct_input_patch, args.pinned_input_dir, args.local_stats_patch,
                                  args.pta_queue_compare))


def run_leg(out, artifacts, workspace, hashes, container, init_pid, args, variant):
    out.mkdir()
    inner_out = "/artifacts/" + out.relative_to(artifacts).as_posix()
    inner_work = "/artifacts/" + workspace.relative_to(artifacts).as_posix()
    flags, options = training_options(args, inner_out)
    env = {"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
           OVERFLOW_SWITCH: "1" if args.queue_strategy or args.extension_patch or variant == "B" else "0"}
    if args.shape_schedule:
        env["TASK_QUEUE_ENABLE"] = "1"
    if args.audit_state:
        env["RECSYS_DYNAMIC_STATE_AUDIT"] = "1"
    if args.queue_strategy:
        env["TASK_QUEUE_ENABLE"] = "2" if variant == "B" else "1"
    if args.pta_queue_compare:
        env["TASK_QUEUE_ENABLE"] = "0" if variant == "B" else "1"
    if args.extension_patch:
        env[args.extension_switch] = "1" if variant == "B" else "0"
    if args.pta_wrapper_module:
        env[args.extension_switch] = "1"
        env["RECSYS_PTA_WRAPPER"] = "1" if args.local_step or variant == "B" or args.pta_control == "eager" else "0"
        env["RECSYS_PTA_REPLAY"] = "1" if args.local_step or variant == "B" else "0"
        env["RECSYS_PTA_SCOPE"] = args.pta_scope
        if args.local_step and (variant == "B" or args.prefetch_patch or args.sidepack_dir):
            env["RECSYS_ZERO2_LOCAL_STEP"] = args.local_step
        if args.prefetch_patch:
            env["RECSYS_CPU_INPUT_PREFETCH"] = str(int(use_cpu_input_prefetch(args, variant)))
        if args.pinned_input_dir:
            env["RECSYS_PINNED_INPUT"] = "1" if variant == "B" else "0"
        if args.direct_input_patch:
            env["RECSYS_HSTU_DIRECT_INPUT"] = "1" if variant == "B" else "0"
        if args.sidepack_dir:
            env["RECSYS_HSTU_GRAPH_SIDEPACK"] = "1" if variant == "B" or args.direct_grad_dir else "0"
        if args.direct_grad_dir:
            env["RECSYS_HSTU_DIRECT_GRADIENT"] = "1" if use_direct_gradient(args, variant) else "0"
        if args.bound_stats_dir:
            env["RECSYS_PTA_BOUND_STATS"] = "1" if variant == "B" else "0"
        if args.local_stats_patch:
            env["RECSYS_PTA_LOCAL_STATISTICS"] = "1" if variant == "B" or any((args.grouped_ops_patch,
                                                          args.joint_stats_patch, args.overlap_prefetch)) else "0"
        if args.graph_loss_patch:
            env["RECSYS_HSTU_GRAPH_LOSS"] = "1" if variant == "B" else "0"
        if args.grouped_ops_patch:
            env["RECSYS_HSTU_GROUPED_SMALL_OPS"] = "1" if variant == "B" else "0"
        if args.joint_stats_patch:
            env["RECSYS_PTA_JOINT_STATS"] = "1" if variant == "B" else "0"
        if args.overlap_input_patch:
            env["RECSYS_OVERLAP_INPUT"] = "1" if variant == "B" or args.graph_guards_patch else "0"
            env["RECSYS_OVERLAP_INPUT_AUDIT"] = str(int(args.overlap_audit))
        if args.graph_guards_patch:
            env["RECSYS_HSTU_FAST_GUARDS"] = "1" if variant == "B" else "0"
    docker_env = [item for name, value in env.items() for item in ("--env", f"{name}={value}")]
    affinity = ["taskset", "-c", args.cpu_affinity] if args.cpu_affinity and variant == "B" else []
    argv = [*DOCKER, "exec", "--workdir", inner_work, *docker_env, container, *affinity,
            "bash", "./run_training.sh", *options]
    write_json(out / "command.json", {"variant": variant, "mode": args.mode, "utc": utc(), "env": env,
                                      "workspace": inner_work, "argv": argv, "shell": shlex.join(argv)})
    code, phase, result = 21, "workspace", None
    try:
        require(files_sha256(workspace) == hashes, "workspace differs from staged snapshot")
        code, phase = 12, "preflight"
        orphan_before = check_card(out, "before", container, init_pid, args.orphan_context_pid)
        write_exit(out / "preflight.exit", 0)
        stop_if_requested()
        code, phase = 20, "training"
        with (out / "run.log").open("wb") as log:
            completed = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, check=False, start_new_session=True)
        write_exit(out / "train.exit", completed.returncode)
        code, phase = 12, "postflight"
        orphan_after = check_card(out, "after", container, init_pid, args.orphan_context_pid)
        require(orphan_before == orphan_after, "orphan HBM context changed within the diagnostic leg")
        write_exit(out / "postflight.exit", 0)
        code, phase = 20, "training"
        require(completed.returncode == 0, f"training exited with {completed.returncode}")
        code, phase = 21, "analysis"
        require(files_sha256(workspace) == hashes, "training changed the staged workspace")
        result = analyze(out, workspace, args, flags, variant)
        write_exit(out / "analysis.exit", 0)
    except (OSError, SyntaxError, ValueError, TypeError, KeyError, subprocess.SubprocessError, RunError) as error:
        if phase in {"preflight", "postflight", "analysis"}:
            write_exit(out / f"{phase}.exit", code)
        if isinstance(error, RunError):
            code = error.status
        write_json(out / "status.json", {"phase": phase, "status": "failed", "detail": str(error), "utc": utc()})
        write_exit(out / "runner.exit", code)
        write_sha256(out / "evidence.sha256", files_sha256(out))
        raise RunError(code, f"{out.name}: {phase}: {error}") from error
    write_json(out / "status.json", {"phase": "analysis", "status": "completed", "utc": utc()})
    write_exit(out / "runner.exit", 0)
    write_sha256(out / "evidence.sha256", files_sha256(out))
    return {"directory": out.name, "variant": variant, "orphan_context": orphan_after, **result}


def run_host(args):
    import fcntl

    require(os.geteuid() == 0, "run on the Linux Docker host as root for complete device process visibility")
    root = args.root.resolve(strict=True)
    artifacts = (root / "artifacts").resolve(strict=True)
    parent = artifacts / STUDY
    parent.mkdir(exist_ok=True)
    require(not parent.is_symlink(), f"unexpected symlink: {parent}")
    prefix = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + f"-{args.stage}-{args.mode}-"
    session = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
    session.chmod(0o755)
    print(session, flush=True)
    write_json(session / "plan.json", {"utc": utc(), "stage": args.stage, "mode": args.mode,
                                       "order": ORDERS[args.stage], "approval_ref": args.approval_ref,
                                       "shape_schedule_sha256": args.shape_schedule_sha256, "audit_state": args.audit_state,
                                       "queue_strategy": args.queue_strategy,
                                       "extension_switch": args.extension_switch,
                                       "extension_patch_sha256": args.extension_patch_sha256,
                                       "extension_module_sha256": args.extension_module_sha256,
                                       "shared_norm_sha256": args.shared_norm_sha256,
                                       "pta_wrapper_sha256": args.pta_wrapper_sha256,
                                       "pta_graph_sha256": args.pta_graph_sha256,
                                       "pta_scope": args.pta_scope if args.pta_wrapper_module else None,
                                       "pta_control": args.pta_control if args.pta_wrapper_module else None,
                                       "local_step": args.local_step, "local_step_sha256": args.local_step_sha256,
                                       "prefetch_patch_sha256": args.prefetch_patch_sha256,
                                       "prefetch_module_sha256": args.prefetch_module_sha256,
                                       "sidepack_sha256": args.sidepack_sha256,
                                       "direct_input_patch_sha256": args.direct_input_patch_sha256,
                                       "direct_input_module_sha256": args.direct_input_module_sha256,
                                       "direct_grad_sha256": args.direct_grad_sha256,
                                       "bound_stats_sha256": args.bound_stats_sha256,
                                       "pinned_input_sha256": args.pinned_input_sha256,
                                       "local_stats_sha256": args.local_stats_sha256,
                                       "pta_queue_compare": args.pta_queue_compare,
                                       "cpu_affinity": args.cpu_affinity,
                                       "graph_loss_sha256": args.graph_loss_sha256,
                                       "grouped_ops_sha256": args.grouped_ops_sha256,
                                       "joint_stats_sha256": args.joint_stats_sha256,
                                       "overlap_input_sha256": args.overlap_input_sha256,
                                       "overlap_module_sha256": args.overlap_module_sha256,
                                       "overlap_audit": args.overlap_audit,
                                       "overlap_prefetch": args.overlap_prefetch,
                                       "graph_guards_sha256": args.graph_guards_sha256,
                                       "orphan_context_pid": args.orphan_context_pid,
                                       "toggle_env": OVERFLOW_SWITCH, "candidate_patch_sha256": args.candidate_patch_sha256,
                                       "candidate_module_sha256": args.candidate_module_sha256})
    code, phase, message, completed = 11, "environment", "", []
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, defer_stop)
    try:
        for tool in ("docker", "npu-smi", "fuser", "git"):
            require(shutil.which(tool), f"missing host tool: {tool}")
        require(stat.S_ISCHR(DEVICE.stat().st_mode), f"{DEVICE} is not a device node")
        with LOCK.open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RunError(10, f"host lock busy: {LOCK}") from error
            container, init_pid = inspect_container(artifacts, session)
            phase = "staging"
            workspace, hashes = prepare(root / "workspace", session, args)
            for name in ("run_npu8.py", "evidence.py"):
                shutil.copyfile(HERE / name, session / "inputs" / name)
            if args.mode != "no-profiler":
                shutil.copyfile(root / "analysis_tools" / "analyze_trace.py", session / "inputs" / "analyze_trace.py")
            phase = "execution"
            for index, variant in enumerate(ORDERS[args.stage], start=1):
                stop_if_requested()
                entry = run_leg(session / f"{index:02d}-{variant}", artifacts, workspace, hashes,
                                container, init_pid, args, variant)
                completed.append(entry)
                if entry["runtime"] != completed[0]["runtime"]:
                    raise RunError(21, "runtime fingerprints differ between legs")
                if entry["orphan_context"] != completed[0]["orphan_context"]:
                    raise RunError(12, "orphan HBM context differs between diagnostic legs")
            stop_if_requested()
            code, phase = 0, "completed"
    except (OSError, ValueError, TypeError, KeyError, SyntaxError, subprocess.SubprocessError, RunError) as error:
        if isinstance(error, RunError):
            code = error.status
        message = str(error)
        print(message, file=sys.stderr)
    finally:
        write_json(session / "status.json", {"phase": phase, "exit": code, "detail": message,
                                             "completed_runs": completed, "utc": utc()})
        write_exit(session / "runner.exit", code)
        write_sha256(session / "evidence.sha256", files_sha256(session))
    return code


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="check staged patches on a temporary copy; no Docker or device access")
    check.add_argument("--workspace", required=True, type=Path)
    run = commands.add_parser("run", help="run an explicitly dispatched experiment on host88")
    run.add_argument("--root", type=Path, default=ROOT)
    run.add_argument("--stage", choices=ORDERS, required=True)
    run.add_argument("--mode", choices=("l0", "l1", "no-profiler"), required=True)
    run.add_argument("--approval-ref", required=True)
    run.add_argument("--shape-schedule", type=Path)
    run.add_argument("--shape-schedule-sha256")
    run.add_argument("--audit-state", action="store_true")
    run.add_argument("--warmup", type=int, default=10)
    run.add_argument("--measure", type=int, default=50)
    run.add_argument("--queue-strategy", action="store_true", help="A/B task queue mode 1/2 with batched overflow in both legs")
    run.add_argument("--orphan-context-pid", type=int, help="diagnostic only: one unnamed device context with no host PID")
    for command in (check, run):
        command.add_argument("--candidate-patch", type=Path)
        command.add_argument("--candidate-module", type=Path)
        command.add_argument("--candidate-patch-sha256")
        command.add_argument("--candidate-module-sha256")
        command.add_argument("--extension-patch", type=Path)
        command.add_argument("--extension-module", type=Path)
        command.add_argument("--extension-patch-sha256")
        command.add_argument("--extension-module-sha256")
        command.add_argument("--extension-switch")
        command.add_argument("--shared-norm-module", type=Path)
        command.add_argument("--shared-norm-sha256")
        command.add_argument("--pta-wrapper-module", type=Path)
        command.add_argument("--pta-wrapper-sha256")
        command.add_argument("--pta-graph-function", type=Path)
        command.add_argument("--pta-graph-sha256")
        command.add_argument("--pta-scope", choices=("both", "norm", "overflow"), default="both")
        command.add_argument("--pta-control", choices=("combined", "eager"), default="combined")
        command.add_argument("--local-step", choices=("flatten", "aliases", "both"))
        command.add_argument("--local-step-module", type=Path)
        command.add_argument("--local-step-sha256")
        command.add_argument("--prefetch-patch", type=Path)
        command.add_argument("--prefetch-patch-sha256")
        command.add_argument("--prefetch-module", type=Path)
        command.add_argument("--prefetch-module-sha256")
        command.add_argument("--sidepack-dir", type=Path)
        command.add_argument("--sidepack-sha256")
        command.add_argument("--direct-input-patch", type=Path)
        command.add_argument("--direct-input-patch-sha256")
        command.add_argument("--direct-input-module", type=Path)
        command.add_argument("--direct-input-module-sha256")
        command.add_argument("--direct-grad-dir", type=Path)
        command.add_argument("--direct-grad-sha256")
        command.add_argument("--bound-stats-dir", type=Path)
        command.add_argument("--bound-stats-sha256")
        command.add_argument("--pinned-input-dir", type=Path)
        command.add_argument("--pinned-input-sha256")
        command.add_argument("--local-stats-patch", type=Path)
        command.add_argument("--local-stats-sha256")
        command.add_argument("--pta-queue-compare", action="store_true",
                             help="compare graph-compatible queue modes 1 and 0 on the prefetch baseline")
        command.add_argument("--cpu-affinity", help="restrict only candidate launch to the given CPU list")
        command.add_argument("--graph-loss-patch", type=Path)
        command.add_argument("--graph-loss-sha256")
        command.add_argument("--grouped-ops-patch", type=Path)
        command.add_argument("--grouped-ops-sha256")
        command.add_argument("--joint-stats-patch", type=Path)
        command.add_argument("--joint-stats-sha256")
        command.add_argument("--overlap-input-patch", type=Path)
        command.add_argument("--overlap-input-sha256")
        command.add_argument("--overlap-input-module", type=Path)
        command.add_argument("--overlap-module-sha256")
        command.add_argument("--overlap-audit", action="store_true")
        command.add_argument("--overlap-prefetch", action="store_true",
                             help="compare overlapped H2D against the CPU worker on the rank-one statistics baseline")
        command.add_argument("--graph-guards-patch", type=Path)
        command.add_argument("--graph-guards-sha256")
    args = parser.parse_args()
    for prefix in ("candidate", "extension"):
        for kind in ("patch", "module"):
            path = getattr(args, f"{prefix}_{kind}")
            digest = getattr(args, f"{prefix}_{kind}_sha256")
            if digest and (not path or not re.fullmatch(r"[0-9a-f]{64}", digest)):
                parser.error(f"{prefix} {kind} SHA-256 requires a file and 64 lowercase hex digits")
    if bool(args.candidate_module) != bool(args.candidate_patch):
        parser.error("candidate requires its training hook patch and overflow module together")
    if args.extension_patch or args.extension_module or args.extension_switch:
        if not (args.candidate_patch and args.extension_patch and args.extension_module and args.extension_switch
                and re.fullmatch(r"RECSYS_(?!ZERO2_BATCH_OVERFLOW$)[A-Z0-9_]+", args.extension_switch)):
            parser.error("extension requires the overflow candidate, extension patch and module, and a separate RECSYS_ switch")
    if args.shared_norm_module or args.shared_norm_sha256:
        if not (args.extension_patch and args.shared_norm_module and args.extension_module.name != "zero2_batch_norm.py"
                and re.fullmatch(r"[0-9a-f]{64}", args.shared_norm_sha256 or "")):
            parser.error("shared norm requires extension and a pinned norm module")
    if any((args.pta_wrapper_module, args.pta_wrapper_sha256, args.pta_graph_function, args.pta_graph_sha256)):
        if not (args.shared_norm_module and args.extension_switch == "RECSYS_HSTU_COMBINED"
                and args.pta_wrapper_module and args.pta_graph_function
                and all(re.fullmatch(r"[0-9a-f]{64}", digest or "")
                        for digest in (args.pta_wrapper_sha256, args.pta_graph_sha256))):
            parser.error("PTA wrapper requires the combined baseline and pinned wrapper and graph modules")
    if any((args.local_step, args.local_step_module, args.local_step_sha256)):
        if not (args.local_step and args.local_step_module and args.pta_wrapper_module
                and args.pta_scope == "both" and args.pta_control == "combined"
                and re.fullmatch(r"[0-9a-f]{64}", args.local_step_sha256 or "")):
            parser.error("local optimizer experiment requires its pinned module and the 08 PTA replay baseline")
    if any((args.prefetch_patch, args.prefetch_module, args.prefetch_patch_sha256, args.prefetch_module_sha256)):
        if not (args.local_step == "both" and args.prefetch_patch and args.prefetch_module and all(
                re.fullmatch(r"[0-9a-f]{64}", digest or "")
                for digest in (args.prefetch_patch_sha256, args.prefetch_module_sha256))):
            parser.error("prefetch requires two pinned inputs and the 09 local optimizer baseline")
    if args.sidepack_dir or args.sidepack_sha256:
        if not (args.local_step == "both" and (not args.prefetch_patch or args.direct_grad_dir) and args.sidepack_dir
                and re.fullmatch(r"[0-9a-f]{64}", args.sidepack_sha256 or "")):
            parser.error("graph sidepack requires its pinned bundle and projects prefetch only onto the direct gradient baseline")
    if any((args.direct_input_patch, args.direct_input_module, args.direct_input_patch_sha256, args.direct_input_module_sha256)):
        if not (args.prefetch_patch and not args.sidepack_dir and args.direct_input_patch and args.direct_input_module and all(
                re.fullmatch(r"[0-9a-f]{64}", digest or "")
                for digest in (args.direct_input_patch_sha256, args.direct_input_module_sha256))):
            parser.error("direct H2D requires two pinned inputs and the 10 prefetch baseline")
    if args.direct_grad_dir or args.direct_grad_sha256:
        if not (args.sidepack_dir and args.direct_grad_dir
                and re.fullmatch(r"[0-9a-f]{64}", args.direct_grad_sha256 or "")):
            parser.error("direct gradients require their pinned bundle and the 11 sidepack replay baseline")
    if args.bound_stats_dir or args.bound_stats_sha256:
        if not (args.direct_grad_dir and not args.prefetch_patch and args.bound_stats_dir
                and re.fullmatch(r"[0-9a-f]{64}", args.bound_stats_sha256 or "")):
            parser.error("bound statistics require their pinned bundle and the 13 direct gradient baseline")
    if args.pinned_input_dir or args.pinned_input_sha256:
        if not (args.prefetch_patch and not args.direct_input_patch and not args.sidepack_dir and args.pinned_input_dir
                and re.fullmatch(r"[0-9a-f]{64}", args.pinned_input_sha256 or "")):
            parser.error("pinned H2D requires its pinned bundle and the 10 prefetch baseline")
    if args.local_stats_patch or args.local_stats_sha256:
        prefetch_baseline = args.prefetch_patch and not args.sidepack_dir
        direct_baseline = args.direct_grad_dir and not args.prefetch_patch
        if not ((prefetch_baseline or direct_baseline) and not any((args.direct_input_patch, args.pinned_input_dir,
                                                                  args.bound_stats_dir, args.cpu_affinity))
                and args.local_stats_patch and re.fullmatch(r"[0-9a-f]{64}", args.local_stats_sha256 or "")):
            parser.error("local statistics require the pinned patch and either the prefetch or direct-gradient baseline")
    if args.cpu_affinity and (not args.direct_grad_dir or any((args.prefetch_patch, args.bound_stats_dir,
            args.local_stats_patch, args.graph_loss_patch)) or not re.fullmatch(r"\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*", args.cpu_affinity)):
        parser.error("CPU affinity requires a valid CPU list and the direct-gradient baseline")
    if args.graph_loss_patch or args.graph_loss_sha256:
        if not (args.direct_grad_dir and not any((args.prefetch_patch, args.bound_stats_dir, args.local_stats_patch,
                                                  args.cpu_affinity)) and args.graph_loss_patch
                and re.fullmatch(r"[0-9a-f]{64}", args.graph_loss_sha256 or "")):
            parser.error("loss capture requires the pinned patch and the direct-gradient baseline")
    if args.grouped_ops_patch or args.grouped_ops_sha256:
        if not (args.prefetch_patch and not args.sidepack_dir and args.local_stats_patch and args.grouped_ops_patch
                and re.fullmatch(r"[0-9a-f]{64}", args.grouped_ops_sha256 or "")):
            parser.error("small op reordering requires the pinned patch and the rank-one statistics prefetch baseline")
    if args.joint_stats_patch or args.joint_stats_sha256:
        if not (args.prefetch_patch and not (args.sidepack_dir or args.grouped_ops_patch)
                and args.local_stats_patch and args.joint_stats_patch
                and re.fullmatch(r"[0-9a-f]{64}", args.joint_stats_sha256 or "")):
            parser.error("joint statistics requires the pinned patch and the rank-one statistics prefetch baseline")
    if any((args.overlap_input_patch, args.overlap_input_sha256, args.overlap_input_module,
            args.overlap_module_sha256, args.overlap_audit, args.overlap_prefetch)):
        direct_baseline = args.direct_grad_dir and not any((args.prefetch_patch, args.bound_stats_dir,
                          args.local_stats_patch, args.cpu_affinity, args.graph_loss_patch, args.overlap_prefetch))
        prefetch_baseline = args.overlap_prefetch and args.prefetch_patch and args.local_stats_patch and not any((
                            args.sidepack_dir, args.grouped_ops_patch, args.joint_stats_patch, args.pta_queue_compare))
        if not ((direct_baseline or prefetch_baseline)
                and args.overlap_input_patch and args.overlap_input_module
                and all(re.fullmatch(r"[0-9a-f]{64}", digest or "") for digest in
                        (args.overlap_input_sha256, args.overlap_module_sha256))):
            parser.error("input overlap requires the pinned patch and a matching direct-gradient or prefetch baseline")
        if args.overlap_audit and args.command == "run" and args.mode != "no-profiler":
            parser.error("input audit requires no-profiler mode")
    if args.graph_guards_patch or args.graph_guards_sha256:
        if not (args.direct_grad_dir and args.overlap_input_patch and args.graph_guards_patch
                and re.fullmatch(r"[0-9a-f]{64}", args.graph_guards_sha256 or "")):
            parser.error("fast graph guards require the pinned patch and the overlapped direct-gradient baseline")
    if args.pta_queue_compare and (not args.prefetch_patch or any((args.direct_input_patch,
            args.pinned_input_dir, args.local_stats_patch, args.sidepack_dir))):
        parser.error("PTA queue comparison uses the 10 prefetch baseline")
    if args.command == "run":
        if args.orphan_context_pid is not None and (not args.local_step or args.orphan_context_pid <= 0):
            parser.error("orphan context is only accepted for explicit local optimizer diagnostics")
        if not args.approval_ref.strip() or min(args.warmup, args.measure) < 1:
            parser.error("approval reference and positive steady-state step counts are required")
        if args.queue_strategy and args.stage not in {"ab", "ba"}:
            parser.error("queue strategy requires an A/B or B/A pair")
        if args.extension_patch and args.stage not in {"ab", "ba", "screen"}:
            parser.error("extension requires an A/B, B/A, or candidate-only screen")
        if args.stage == "screen" and not args.extension_patch:
            parser.error("candidate-only screen requires an extension")
        if args.extension_patch and (args.queue_strategy or not (args.extension_patch_sha256 and args.extension_module_sha256)):
            parser.error("extension requires approved SHA-256 values and queue strategy unchanged")
        if args.stage in {"ab", "ba", "screen"}:
            if not (args.candidate_patch_sha256 and args.candidate_module_sha256):
                parser.error("A/B requires approved candidate patch and module with SHA-256")
        elif args.candidate_patch:
            parser.error("baseline and A/A use only the common steady-state patch")
    return args


def main():
    args = parse_args()
    if args.command == "run":
        if not sys.platform.startswith("linux"):
            print("run requires the Linux Docker host; use check for offline validation", file=sys.stderr)
            return 11
        try:
            return run_host(args)
        except (OSError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 11
    with tempfile.TemporaryDirectory(prefix="host88-npu8-check-") as tmp:
        session = Path(tmp)
        try:
            _, hashes = prepare(args.workspace, session, args)
        except (OSError, ValueError, KeyError, SyntaxError, subprocess.SubprocessError) as error:
            print(str(error), file=sys.stderr)
            log = session / "patches.log"
            if log.is_file():
                print(log.read_text(encoding="utf-8"), file=sys.stderr, end="")
            return 11
        print((session / "patches.log").read_text(encoding="utf-8"), end="")
        print(f"Patch and Python syntax checks passed for {len(hashes)} staged files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
