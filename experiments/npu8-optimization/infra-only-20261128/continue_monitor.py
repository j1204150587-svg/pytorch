"""Extend the existing device sampler without inserting a gap at its time limit."""
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
prior_pid=int(sys.argv[1])
old=Path("/proc")/str(prior_pid)/"cmdline"
while old.exists() and b"chip_occupancy.py" in old.read_bytes():
    if (root/"monitor-complete").exists():
        sys.exit(0)
    time.sleep(1)
if (root/"monitor-complete").exists():
    sys.exit(0)
source=root.parent/"dynamic-infra-20261127/dynamic-free70/runner_tools/chip_occupancy.py"
impl=source.read_text()
orig='with args.output.open("x",buffering=1) as log:'
assert impl.count(orig)==1
impl=impl.replace(orig,'with args.output.open("a",buffering=1) as log:').replace(
    "while time.monotonic()<end:",'while time.monotonic()<end and not (args.output.parent/"monitor-complete").exists():')
sys.argv=[str(source),"--output",str(root/"chip.jsonl"),"--minutes","180"]
exec(compile(impl,str(source),"exec"),{"__name__":"__main__"})

