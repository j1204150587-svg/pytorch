"""Read scheduler events filtered in-kernel to one verified training consumer TID."""
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
from decimal import Decimal


def write(path, value):
    path.write_text(str(value) + "\n")


def get_local_tid(path):
    fields = dict(line.split(":", 1) for line in (path / "status").read_text().splitlines() if ":" in line)
    return int(fields["NSpid"].split()[-1])


def find_consumer(docker, container, workspace, target):
    top = subprocess.check_output([*docker, "top", container, "-eo", "pid"], text=True)
    pids = [Path("/proc") / x.strip() for x in top.splitlines() if x.strip().isdigit()]
    matches = []
    for process in pids:
        try:
            if get_local_tid(process) != target["pid"] or not os.path.samefile(process / "cwd", workspace):
                continue
            for task in (process / "task").iterdir():
                if (task / "comm").read_text().strip() == "acl_thread" and get_local_tid(task) == target["tid"]:
                    matches.append((int(process.name), int(task.name)))
        except (FileNotFoundError, ProcessLookupError):
            continue
    if len(matches) != 1:
        raise ValueError(f"expected one matching container/workspace/TID, found {len(matches)}")
    return matches[0]


stamp = re.compile(r"\s(\d+\.\d+):\s+(sched_switch|sched_wakeup):\s+(.*)$")


def decode(line, tid):
    match = stamp.search(line)
    if match is None:
        if not line.strip() or line.startswith("#"):
            return None
        raise ValueError("unexpected trace_pipe record or lost events")
    value, event, fields = match.groups()
    timestamp = int(Decimal(value) * 1_000_000_000)
    if event == "sched_wakeup":
        if not re.search(rf"\bpid={tid}\b", fields):
            raise ValueError("non-target wakeup")
        return timestamp, "wake", ""
    if re.search(rf"\bprev_pid={tid}\b", fields):
        state = re.search(r"\bprev_state=(\S+)", fields)
        if state is None:
            raise ValueError("missing outgoing thread state")
        return timestamp, "out", state.group(1)
    if re.search(rf"\bnext_pid={tid}\b", fields):
        return timestamp, "in", ""
    raise ValueError("non-target switch")


def losses(instance):
    totals = {"overrun": 0, "commit overrun": 0, "dropped events": 0}
    for path in (instance / "per_cpu").glob("cpu*/stats"):
        for line in path.read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() in totals:
                totals[key.strip()] += int(value.strip())
    return totals


def trace_one(instance, directory, tid, training):
    write(instance / "tracing_on", 0)
    if "mono_raw" not in (instance / "trace_clock").read_text().replace("[", "").replace("]", "").split():
        raise ValueError("private tracefs instance lacks mono_raw")
    write(instance / "trace_clock", "mono_raw")
    if "[mono_raw]" not in (instance / "trace_clock").read_text():
        raise ValueError("mono_raw was not selected")
    write(instance / "buffer_size_kb", 256)
    nsecs = instance / "options/nsecs"
    if nsecs.exists():
        write(nsecs, 1)
    for event, condition in (("sched_switch", f"prev_pid == {tid} || next_pid == {tid}"),
                             ("sched_wakeup", f"pid == {tid}")):
        event_path = instance / "events/sched" / event
        write(event_path / "filter", condition)
        if (event_path / "filter").read_text().strip() != condition:
            raise ValueError(f"could not verify private {event} filter")
        write(event_path / "enable", 1)
    start_loss = losses(instance)
    raw = os.open(instance / "trace_pipe", os.O_RDONLY | os.O_NONBLOCK)
    pending = b""
    resolution = 1
    count = 0
    try:
        with (directory / "scheduler.csv").open("w", newline="") as stream:
            out = csv.writer(stream)
            out.writerow(["mono_raw_ns", "event", "prev_state"])

            def drain():
                nonlocal pending, resolution, count
                while True:
                    try:
                        block = os.read(raw, 262144)
                    except BlockingIOError:
                        break
                    if not block:
                        break
                    pending += block
                    lines = pending.split(b"\n")
                    pending = lines.pop()
                    for data in lines:
                        line = data.decode("utf-8", "replace")
                        record = decode(line, tid)
                        if record is not None:
                            precision = len(stamp.search(line).group(1).split(".")[1])
                            resolution = max(resolution, 10 ** max(0, 9 - precision))
                            out.writerow(record)
                            count += 1
            write(instance / "tracing_on", 1)
            (directory / "ready").touch()
            while training.poll() is None and not (directory / "finished").exists():
                drain()
                time.sleep(.01)
            write(instance / "tracing_on", 0)
            drain()
            if pending:
                raise ValueError("partial trace record")
    finally:
        os.close(raw)
    end_loss = losses(instance)
    if count == 0 or any(end_loss[k] != start_loss[k] for k in end_loss):
        raise ValueError(f"incomplete events, count={count}, loss {start_loss} -> {end_loss}")
    return dict(count=count, trace_resolution_ns=resolution, loss_before=start_loss, loss_after=end_loss)


def run_target(argv, log, out, workspace, docker, container):
    directory = out / "rank000/queue_sched"
    roots = (Path("/sys/kernel/tracing/instances"), Path("/sys/kernel/debug/tracing/instances"))
    parent = next((p for p in roots if p.is_dir()), None)
    if parent is None:
        raise ValueError("tracefs private instances unavailable")
    training = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    instance = parent / ("hstu8_" + uuid.uuid4().hex[:12])
    try:
        while training.poll() is None and not (directory / "target.json").is_file():
            time.sleep(.05)
        if training.poll() is None:
            target = json.loads((directory / "target.json").read_text())
            pid, tid = find_consumer(docker, container, workspace, target)
            instance.mkdir()
            meta = trace_one(instance, directory, tid, training)
            meta.update(host_pid=pid, host_tid=tid, **target, clock="mono_raw")
            (directory / "scheduler.json").write_text(json.dumps(meta, indent=2))
    except Exception as error:
        if directory.is_dir():
            (directory / "error").write_text(str(error))
        # The staged process checks the error during its handshake. The normal runner
        # keeps the card lock until the training command and profiler export finish.
        training.wait()
        raise
    finally:
        if instance.exists():
            write(instance / "tracing_on", 0)
            instance.rmdir()
    return subprocess.CompletedProcess(argv, training.wait())
