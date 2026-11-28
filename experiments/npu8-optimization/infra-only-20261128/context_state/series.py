"""Run the new cache by itself on the verified baseline, under the existing card ownership protocol."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    here=Path(__file__).resolve().parent
    root=here.parent
    assert len(json.loads((here/"mock/mock.json").read_bytes())["records"])==2
    for compatible in "01":
        for v in "01":
            micro=here/f"micro-{compatible}-{v}"
            while not (micro/"after.card.json").is_file():time.sleep(5)
            assert (micro/"train.exit").read_text().strip()=="0"
            p=json.loads((micro/"micro.json").read_bytes())
            assert p["switch"]==v and p["compatible"]==compatible and p["positive_reads"]==2048
    for mode,order in (("audit","ab"),("no-profiler","ab"),("no-profiler","ba"),
                       ("l0","ab"),("l0","ba"),("l1","ab")):
        dest=root/"pairs"/f"context-state-{mode}-{order}"
        if not (dest/"result.json").exists():
            suffix="-audit" if mode=="audit" else ""
            command=[sys.executable,"-B","runner/host_pair.py","--runner","archived-runner",
                "--recipe",f"recipes/context-state{suffix}.json","--out",str(dest),"--mode",mode,"--order",order]
            print(" ".join(command),flush=True)
            subprocess.run(command,cwd=root,check=True)
        result=json.loads((dest/"result.json").read_bytes())
        assert result["exit"]==0
        if mode=="audit":
            assert result["audit_passed"] and result["audited_steps"]==8
            counts=result["native_counters"]
            a,b=[counts[k]["npu_active_context_counter"] for k in "AB"]
            assert a["accepted"]==a["keys"]==0 and a["fallback"]>0
            assert b["accepted"]>0 and b["keys"]>0 and b["accepted"]>b["fallback"] and not b["descriptors"],b
            for v in "AB":
                for d in ("forward","backward"):
                    assert counts[v][f"npu_embedding_{d}_rebase_counter"]["accepted"]==8192
        print("VERIFIED",dest,flush=True)

if __name__=="__main__":
    main()
