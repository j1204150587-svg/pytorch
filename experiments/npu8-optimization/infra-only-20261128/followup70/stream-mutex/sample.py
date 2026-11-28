"""Collect short, rank-scoped lock diagnostics on an unchanged, shared-lock A/A run."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CONTAINER = "recsys-gr-ds019-npu8"
GROUP = "hstu_npu8_stream_20261128"
TRACE = Path("/sys/kernel/tracing")
ELF = "usr/local/Ascend/cann-9.1.0/arm64-linux/lib64/libruntime_v100.so"
EXPECTED = "ab4b573989e6b76e129204cf329bdd916739bdd16092a0efcfa88d8ba011d689"
# Shipping v100 passes the mutex address in x0 immediately before calling libc.
# At the post-call sites x0 is the glibc return code. All probes precede the instruction.
PROBES = {
    "s_begin": (0x379EA8, "addr=%x0:x64 word=+0(%x0):u32 caller=%x30:x64"),
    "s_got":   (0x379EAC, "rc=%x0:u32"),
    "s_end":   (0x379ED0, "addr=%x0:x64 word=+0(%x0):u32 caller=%x30:x64"),
    "r_begin": (0x37A010, "addr=%x0:x64 word=+0(%x0):u32 caller=%x30:x64"),
    "r_got":   (0x37A014, "rc=%x0:u32"),
    "r_end":   (0x37A038, "addr=%x0:x64 word=+0(%x0):u32 caller=%x30:x64"),
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def tracer(line):
    # tracefs accepts O_APPEND writes but does not support Python's initial append seek.
    fd = os.open(TRACE / "uprobe_events", os.O_WRONLY | os.O_APPEND)
    try:
        os.write(fd, (line + "\n").encode())
    finally:
        os.close(fd)


def collect(leg, pid, rootfs):
    proc = Path("/proc") / str(pid)
    mapping = [line for line in (proc / "maps").read_text().splitlines()
               if line.endswith(("libruntime_v100.so", "libc.so.6"))]
    assert any("r-xp" in row and row.endswith("libruntime_v100.so") for row in mapping)
    tids = {}
    for task in (proc / "task").iterdir():
        try:
            tids[task.name] = (task / "comm").read_text().strip()
        except FileNotFoundError:
            pass
    write(HERE / (leg + ".worker.json"), dict(pid=pid, maps=mapping, tids=tids))
    commands = []
    probe_args = [entry for name in PROBES for entry in ("-e", GROUP + ":" + name)]
    for name in ("sys_enter_futex", "sys_exit_futex"):
        probe_args.extend(["-e", "syscalls:" + name])
    cmd = ["perf", "record", "-m", "128", "-o", str(HERE / (leg + ".locks.data")),
           *probe_args, "-p", str(pid), "--", "sleep", "0.65"]
    commands.append(cmd)
    subprocess.run(cmd, check=True)
    # Disable probe events at the end of perf record before measuring memory traffic.
    cmd = ["perf", "record", "-m", "64,128", "-c", "4096", "-e",
           "arm_spe_0/load_filter=1,store_filter=1,jitter=1,min_latency=0/u",
           "-o", str(HERE / (leg + ".spe.data")), "-p", str(pid), "--", "sleep", "0.7"]
    commands.append(cmd)
    write(HERE / (leg + ".commands.json"), commands)
    subprocess.run(cmd, check=True)
    assert proc.is_dir()
    with (HERE / (leg + ".locks.txt")).open("w") as out:
        subprocess.run(["perf", "script", "--ns", "-i", str(HERE / (leg + ".locks.data")),
                        "-F", "trace:comm,pid,tid,cpu,time,event,trace"],
                       stdout=out, check=True)


def find_rank(leg):
    dispatch = HERE / "pair" / leg / "dispatch.json"
    deadline = time.monotonic() + 3600
    while time.monotonic() < deadline:
        if dispatch.exists():
            info = json.loads(dispatch.read_bytes())
            log = Path(info["session"]) / leg / "run.log"
            if log.exists() and "VariableFallbackKernel" in log.read_text(errors="replace"):
                break
        time.sleep(0.25)
    else:
        raise TimeoutError(leg)
    processes = subprocess.check_output(["docker", "top", CONTAINER, "-eo", "pid"], text=True)
    pids = []
    for number in processes.splitlines()[1:]:
        proc = Path("/proc") / number.strip()
        try:
            if (info["output"].encode() in (proc / "cmdline").read_bytes().split(b"\0")
                    and b"LOCAL_RANK=0" in (proc / "environ").read_bytes().split(b"\0")):
                pids.append(int(number))
        except FileNotFoundError:
            pass
    assert len(pids) == 1
    time.sleep(3)
    return pids[0]


def main():
    init = subprocess.check_output(["docker","inspect","-f","{{.State.Pid}}", CONTAINER], text=True).strip()
    rootfs = Path("/proc") / init / "root"
    binary = rootfs / ELF
    sha = subprocess.check_output(["sha256sum", binary], text=True).split()[0]
    assert sha == EXPECTED
    added = []
    try:
        for name, (offset, args) in PROBES.items():
            entry = f"p:{GROUP}/{name} {binary}:0x{offset:x} {args}"
            tracer(entry)
            added.append(name)
        assert len(added) == 6
        if "--check" in sys.argv:
            print("registered six private events")
            return
        for leg in ("01-A", "02-B"):
            collect(leg, find_rank(leg), rootfs)
    finally:
        for name in reversed(added):
            tracer(f"-:{GROUP}/{name}")


if __name__ == "__main__":
    main()
