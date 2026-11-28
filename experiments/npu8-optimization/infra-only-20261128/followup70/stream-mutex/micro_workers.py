"""Record only the four private child workers that belong to this diagnostic."""
import json
from pathlib import Path
import subprocess
import sys
import time


here=Path(__file__).resolve().parent
out=here/sys.argv[1]
workers={}
until=time.monotonic()+3600
while not (out/"train.exit").exists() and time.monotonic()<until:
    if not (out/"command.json").exists():
        time.sleep(1)
        continue
    lines=subprocess.check_output(["docker","top","recsys-gr-ds019-npu8","-eo","pid"],text=True).splitlines()[1:]
    for value in lines:
        pid=int(value.strip())
        proc=Path("/proc")/str(pid)
        try:
            args=(proc/"cmdline").read_bytes().decode().split("\0")
            if "--arm" not in args or not any(s.endswith("/stream-mutex/micro.py") for s in args): continue
            part=next(s for s in args if s.endswith(".json") and f"/stream-mutex/{out.name}/part-" in s)
            ns=next(x for x in (proc/"status").read_text().splitlines() if x.startswith("NSpid:")).split()
            workers[Path(part).name]={"host_pid":pid,"container_pid":int(ns[-1]),"arm":args[args.index("--arm")+1]}
        except (FileNotFoundError,ProcessLookupError,StopIteration):
            pass
    (out/"workers.json").write_text(json.dumps(workers,indent=2)+"\n")
    time.sleep(.45)
