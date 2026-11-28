"""Continue after the queued micro, preserving exit conditions and the stock pair lock."""
from pathlib import Path
import json
import subprocess
import sys
import time


if __name__=="__main__":
    base=Path(__file__).resolve().parent
    while not (base/"micro-v2.exit").exists():
        time.sleep(10)
    if (base/"micro-v2.exit").read_text().strip()!="0":
        raise SystemExit("micro gate failed")
    c=json.loads((base/"micro-run-v2/micro.json").read_bytes())["cases"]
    a,b,bb,aa=[x["median_us"] for x in c[-4:]]
    diffs={tag:[b[tag]-a[tag],bb[tag]-aa[tag]] for tag in a}
    action="continue" if any(any(delta<0 for delta in values) for values in diffs.values()) else "stop_no_micro_gain"
    (base/"performance-gate.json").write_text(json.dumps({"b_minus_a_us":diffs,"action":action},indent=2))
    if action=="continue":
        for stage in ("audit","formal"):
            subprocess.run([sys.executable,"-B",str(base/"run_locked.py"),stage],check=True)
