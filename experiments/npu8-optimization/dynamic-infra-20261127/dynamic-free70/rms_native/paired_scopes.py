"""Link the nine RMS forward/backward scopes to official Free and physical tasks on four Level0 legs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"rms_full"))
from scope_counts import count


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for label in ("ab_A","ab_B","ba_A","ba_B"):
        p.add_argument("--"+label,required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args()
    data={}
    for label,trace in vars(args).items():
        if label=="out":continue
        data[label]={"trace_sha256":hashlib.sha256(trace.read_bytes()).hexdigest(),
                     "steps":count(trace,native=label.endswith("B"))}
    args.out.write_text(json.dumps(data,indent=2)+"\n")
