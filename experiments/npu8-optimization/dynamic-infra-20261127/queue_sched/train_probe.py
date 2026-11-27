"""Target-thread handshake and monotonic-raw anchors in a diagnostic training process."""
import json
import os
from pathlib import Path
import time


def prepare(directory):
    directory.mkdir()
    threads = [p for p in Path("/proc/self/task").iterdir() if (p / "comm").read_text().strip() == "acl_thread"]
    if len(threads) != 1:
        raise RuntimeError(f"need one acl_thread, found {len(threads)}")
    target = dict(pid=os.getpid(), tid=int(threads[0].name), comm="acl_thread")
    temp = directory / "target.tmp"
    temp.write_text(json.dumps(target))
    temp.replace(directory / "target.json")
    deadline = time.monotonic() + 90
    while not (directory / "ready").exists():
        if (directory / "error").exists() or time.monotonic() >= deadline:
            raise RuntimeError("private scheduler capture did not arm")
        time.sleep(.02)


def clock_anchors(directory, record_function, step):
    raw = time.CLOCK_MONOTONIC_RAW
    samples = []
    for attempt in range(5):
        name = f"pta_sched_clock_{step}_{attempt}"
        marker = record_function(name)
        before = time.clock_gettime_ns(raw)
        with marker:
            inside_start = time.clock_gettime_ns(raw)
            inside_end = time.clock_gettime_ns(raw)
        after = time.clock_gettime_ns(raw)
        samples.append([name, before, inside_start, inside_end, after])
    with (directory / "clock.jsonl").open("a") as handle:
        handle.write(json.dumps(samples) + "\n")
