"""Validate the new NPU 8 run artifacts with no hardware or vendor imports."""

import csv
import hashlib
import json
import math
import re


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def files_sha256(root, exclude=()):
    hashes = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if relative in exclude:
            continue
        require(not path.is_symlink(), f"unexpected symlink: {path}")
        if path.is_file():
            hashes[relative] = sha256(path)
    return hashes


def write_sha256(path, hashes):
    path.write_text("".join(f"{digest}  {name}\n" for name, digest in hashes.items()), encoding="utf-8")


def header_index(rows, heading):
    matches = [i for i, row in enumerate(rows) if row == heading]
    require(len(matches) == 1, f"expected one npu-smi header: {heading}")
    return matches[0]


def npu_mapping(report, logic_id):
    rows = [" ".join(line.split()) for line in report.splitlines() if line.strip()]
    start = header_index(rows, "NPU ID Chip ID Chip Logic ID Chip Phy-ID Chip Name")
    matches, compute_chips = [], set()
    for row in rows[start + 1:]:
        match = re.fullmatch(r"(\d+) (\d+) (\d+|-) (\d+|-) (\S+)", row)
        require(match is not None, f"unparsed npu-smi mapping row: {row}")
        board, chip, logical, physical, _ = match.groups()
        if logical == physical == "-":
            continue  # Mcu rows have no logic or physical ID.
        require(logical.isdecimal() and physical.isdecimal(), f"incomplete chip mapping: {row}")
        pair = (int(board), int(chip))
        require(pair not in compute_chips, f"duplicate npu-smi mapping for {pair}")
        compute_chips.add(pair)
        if int(logical) == logic_id:
            matches.append((*pair, int(physical)))
    require(len(matches) == 1, f"cannot map logical device {logic_id} uniquely with npu-smi info -m")
    return matches[0], compute_chips


def table_fields(line):
    if not line or re.fullmatch(r"\+[-=+]+\+", line):
        return ()
    require(line.startswith("|") and line.endswith("|"), f"unparsed npu-smi table row: {line}")
    return tuple(" ".join(field.split()) for field in line[1:-1].split("|"))


def idle_card(report, mapping_report, logic_id, orphan_pid=None):
    """Confirm target health and process-table occupancy using the live logical-ID mapping."""
    (npu, chip, physical), compute_chips = npu_mapping(mapping_report, logic_id)
    lines = [line.strip() for line in report.splitlines() if line.strip()]
    fields = [table_fields(line) for line in lines]
    headers = [row[:2] for row in fields]
    chip_header = header_index(headers, ("Chip Phy-ID", "Bus-Id"))
    process_header = header_index(fields, ("NPU Chip", "Process id", "Process name", "Process memory(MB)"))
    require(chip_header < process_header, "npu-smi tables out of order")
    require(not fields[-1], "npu-smi process table has no closing separator")

    status_rows = [(i, row) for i, row in enumerate(fields) if chip_header < i < process_header and row]
    require(len(status_rows) % 2 == 0, "npu-smi chip status table has an incomplete pair")
    matches = []
    for (board_index, board_row), (chip_index, chip_row) in zip(status_rows[::2], status_rows[1::2]):
        board = re.fullmatch(r"(\d+) [A-Za-z][\w-]*", board_row[0])
        ids = re.fullmatch(r"(\d+) (\d+)", chip_row[0])
        require(board and ids and len(board_row) == len(chip_row) == 3 and board_row[1]
                and re.fullmatch(r"[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.\d", chip_row[1]),
                f"unparsed npu-smi chip status pair: {lines[board_index]} / {lines[chip_index]}")
        found = (int(board[1]), int(ids[1]), int(ids[2]))
        if found[:2] == (npu, chip) or found[2] == physical:
            matches.append((found, board_row[1], lines[chip_index]))
    require(len(matches) == 1 and matches[0][0] == (npu, chip, physical),
            f"npu-smi status disagrees with logical device {logic_id} mapping {(npu, chip, physical)}")
    _, health, device_row = matches[0]
    require(health == "OK", f"NPU {npu} chip {chip} health is {health}")

    empty_boards, board_processes, orphan_context = set(), [], None
    for i in range(process_header + 1, len(fields)):
        row = fields[i]
        if not row:
            continue
        empty = re.fullmatch(r"No running processes found in NPU (\d+)", row[0])
        if len(row) == 1 and empty:
            empty_boards.add(int(empty[1]))
            continue
        ids = re.fullmatch(r"(\d+) (\d+)", row[0])
        require(len(row) == 4 and ids and row[1].isdecimal() and int(row[1]) > 0
                and row[3].isdecimal(), f"unparsed npu-smi process row: {lines[i]}")
        board_id, chip_id = map(int, ids.groups())
        require((board_id, chip_id) in compute_chips, f"process row has an unmapped chip: {lines[i]}")
        pid = int(row[1])
        if not row[2]:
            require(pid == orphan_pid and (board_id, chip_id) == (npu, chip) and orphan_context is None,
                    f"unnamed device context requires diagnosis: {lines[i]}")
            orphan_context = {"pid": pid, "memory_mb": int(row[3])}
            continue
        if board_id == npu:
            board_processes.append({"chip": chip_id, "pid": pid, "name": row[2], "memory_mb": int(row[3])})
    active = [row["pid"] for row in board_processes if row["chip"] == chip]
    require(not active, f"logical device {logic_id} (NPU {npu} chip {chip}) is occupied by host PIDs {active}")
    require((npu in empty_boards) != bool(board_processes or orphan_context),
            f"missing or conflicting process evidence for NPU {npu}")
    return {"logical_device": logic_id, "board": npu, "chip": chip, "physical_id": physical, "health": health,
            "device_row_with_memory": device_row, "host_pids": [], "same_board_other_chip_processes": board_processes,
            "orphan_context": orphan_context}


def validate_training(out, workspace, mode, flags):
    rank = out / "rank000"
    arguments = read_json(rank / "run_arguments.json")
    runtime = read_json(rank / "runtime.json")
    summary = read_json(rank / "model_summary.json")
    expected_model = read_json(workspace / flags["model_config"])
    expected_ds = read_json(workspace / flags["deepspeed_config"])
    expected_ds.update(train_micro_batch_size_per_gpu=flags["micro_batch_size"],
                       train_batch_size=flags["micro_batch_size"] * expected_ds["gradient_accumulation_steps"])
    effective_ds = read_json(rank / "effective_deepspeed_config.json")
    require(effective_ds == expected_ds and effective_ds["bf16"]["check_grad_overflow"] is True,
            "unexpected effective DeepSpeed configuration")
    require(summary["config"] == expected_model and expected_model["num_tables"] == 1024
            and summary["trainable_parameter_tensors"] == 1052
            and summary["trainable_parameter_elements"] == 4380544, "unexpected workload")
    require(all(arguments.get(key) == value for key, value in flags.items()), "unexpected training arguments")
    require((runtime["rank"], runtime["world_size"], runtime["local_rank"], runtime["device"]) == (0, 1, 0, "npu:0"),
            "expected one training rank on container npu:0")
    document = read_json(rank / "metrics.json")
    metrics, rows = document["summary"], document["steps"]
    steps, batch, log_interval = (flags[key] for key in ("steps", "micro_batch_size", "log_interval"))
    require(metrics["status"] == "completed" and metrics["steps"] == steps and metrics["rank"] == 0
            and metrics["profile_enabled"] is (mode != "no-profiler"), "incomplete training summary")
    require([row["step"] for row in rows] == list(range(steps)), "missing or duplicate training steps")
    log_steps = [step for step in range(steps) if step == 0 or (step + 1) % log_interval == 0]
    require([row["step"] for row in rows if "loss" in row] == log_steps, "unexpected loss log cadence")
    require(all(math.isfinite(rows[step]["loss"]) for step in log_steps), "non-finite logged loss")
    reported = {"runtime": runtime, "loss_steps": log_steps}
    if mode == "no-profiler":
        require(not (rank / "profile").exists(), "unexpected profiler artifacts in no-profiler mode")
        steady = metrics["steady_state"]
        warmup, measured = flags["steady_warmup_steps"], flags["steady_measure_steps"]
        require((steady["start_step_inclusive"], steady["end_step_exclusive"], steady["steps"], steady["samples"])
                == (warmup, steps, measured, measured * batch), "unexpected steady-state window")
        elapsed = steady["elapsed_s_synchronized"]
        require(math.isfinite(elapsed) and 0 < elapsed <= metrics["elapsed_s_synchronized"], "invalid steady elapsed")
        require(math.isclose(steady["samples_per_s"], measured * batch / elapsed)
                and math.isclose(steady["ms_per_step"], elapsed * 1000 / measured)
                and steady["loss_log_policy"] == "deferred_until_after_end_sync", "inconsistent steady metrics")
        reported["steady_state"] = steady
    else:
        require("steady_state" not in metrics, "unexpected steady metric from a profiled run")
    return reported


def validate_step_csv(path, active):
    with path.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    fields = ("Stage", "Free", "Computing", "Communication(Not Overlapped)")
    require([int(row["Step"]) for row in rows] == list(range(4, 4 + active)), "unexpected profile step IDs")
    require(len({row["Device_id"] for row in rows}) == 1, "profile contains more than one device")
    totals = {key: sum(float(row[key]) for row in rows) for key in fields}
    require(all(math.isfinite(value) and value >= 0 for value in totals.values()) and totals["Stage"] > 0,
            "invalid profiler durations")
    return {"steps": [int(row["Step"]) for row in rows], "device_id": rows[0]["Device_id"],
            "mean_ms": {key: value / active / 1000 for key, value in totals.items()},
            "free_over_stage": totals["Free"] / totals["Stage"]}


def validate_overflow_scan(path, variant, steps):
    if variant == "A":
        require(not path.exists(), "A produced candidate overflow evidence despite toggle=0")
        return None
    require(path.is_file(), "B did not write rank000/overflow_scan.json")
    stats = read_json(path)
    require(stats["training_completed"] is True and stats["valid"] is True
            and (stats["expected_steps"], stats["calls"], stats["fast_calls"]) == (steps, steps, steps)
            and stats["fallback_calls"] == stats["errors"] == 0 and not stats["reasons"]
            and stats["chunks"] >= steps,
            "B did not run the fast scan on every training step")
    return stats
