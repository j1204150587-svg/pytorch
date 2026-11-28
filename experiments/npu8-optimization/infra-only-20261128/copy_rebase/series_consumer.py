"""Schedule native copy-preparation micro, gradient audit and crossed timing/profile pairs."""
import json
from pathlib import Path
import subprocess
import sys


def main():
    here=Path(__file__).resolve().parent
    root=here.parent
    inner="/artifacts/infra-only-20261128"
    def run(*command):
        print(" ".join(map(str,command)),flush=True)
        subprocess.run(list(map(str,command)),cwd=root,check=True)
    for toggle in ("0","1"):
        dest=here/"runs-consumer"/("micro-"+toggle)
        if not (dest/"micro.json").exists():
            run(sys.executable,"-B","operator_probe.py","--package",inner+"/stock-overlay/copy-consumer",
                "--env","ACLNN_CACHE_LIMIT=65536","--env","NPU_DYNAMIC_REBASE_INDICES=1",
                "--env","NPU_DYNAMIC_REBASE_FORWARD=1",dest,inner+"/copy_rebase/micro_consumer.py",
                "@OUT/micro.json","--switch",toggle)
        summary=json.loads((dest/"micro.json").read_bytes())
        assert summary["overlap_error_preserved"] and len(summary["checks"])==14,summary
    for mode,order in (("audit","ab"),("no-profiler","ab"),("no-profiler","ba"),("l0","ab"),("l0","ba"),("l1","ab")):
        out=root/"pairs"/f"copy-consumer-{mode}-{order}"
        if not (out/"result.json").exists():
            suffix="-audit" if mode=="audit" else ""
            run(sys.executable,"-B","runner/host_pair.py","--runner","archived-runner",
                "--recipe",f"recipes/copy-consumer{suffix}.json","--out",out,"--mode",mode,"--order",order)
        result=json.loads((out/"result.json").read_bytes())
        assert result["exit"]==0
        if mode=="audit":
            assert result["audit_passed"] and result["audited_steps"]==8
            counts=result["native_counters"]["B"]["npu_inplace_copy_rebase_counter"]
            assert counts["accepted"]==counts["keys"]==counts["descriptors"] and counts["accepted"]>0,counts
        print("VERIFIED",out,flush=True)


if __name__=="__main__":
    main()
