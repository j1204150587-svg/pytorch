"""Bracket host raw counter against wall time without attaching to any worker."""
import json
from pathlib import Path
import time

here=Path(__file__).resolve().parent
end=time.monotonic()+3*3600
with (here/"clock-samples.jsonl").open("a",buffering=1) as fp:
    while time.monotonic()<end and not (here/"clock.stop").exists():
        batch=[]
        for _ in range(5):
            left=time.clock_gettime_ns(time.CLOCK_REALTIME)
            raw=time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
            right=time.clock_gettime_ns(time.CLOCK_REALTIME)
            batch.append((right-left,left,raw,right))
        _,left,raw,right=min(batch)
        print(json.dumps([left,raw,right]),file=fp)
        time.sleep(.05)
