"""Run the existing tiny ZeRO-2 skip/recovery diagnostic with both CANN instruction paths."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("workspace",type=Path)
    ap.add_argument("out",type=Path)
    a=ap.parse_args()
    here=Path(__file__).resolve().parent
    original=here.parents[1]/"copy_rebase/overflow_probe.py"
    original_hash=hashlib.sha256(original.read_bytes()).hexdigest()
    if original_hash!="d1e2331f794c9856993bbd1bfdb182ddbbaa4094f1699364697ae5dee86b37fd":
        raise ValueError(original_hash)
    code=original.read_text()
    old='if r.split()[-1].endswith("/libtorch_npu.so")}'
    assert code.count(old)==1
    code=code.replace(old,'if r.split()[-1].endswith("/libruntime_v100.so")}\n    assert library==os.environ["EXPECTED_PENDING_RUNTIME"]')
    code=code.replace("ctypes.CDLL(library).npu_inplace_copy_rebase_counter","ctypes.CDLL(library).npu_cann_pending_count")
    code=code.replace("        before_count=counter(0)","        torch.npu.synchronize()\n        lane=int(args.switch)\n        before_count=counter(lane)")
    code=code.replace("        accepted=counter(0)-before_count\n        assert accepted==int(args.switch), (i,args.flag,accepted)",
                      "        torch.npu.synchronize()\n        accepted=counter(lane)-before_count\n        assert accepted>0 and counter(1-lane)==0 and counter(3)==lane")
    code=code.replace("target_copy_accepted=accepted","pending_submissions_while_copying=accepted")
    src=a.out/"zero2_program.py"
    src.write_text(code)
    runtime=here/"source-v2/libruntime_v100.so"
    rows=[]
    for lane in (0,1):
        dest=a.out/f"zero2-{lane}.json"
        env=dict(os.environ,NPU_CANN_PENDING_LSE=str(lane),NPU_CANN_PENDING_COUNT="1",EXPECTED_PENDING_RUNTIME=str(runtime),
                 LD_LIBRARY_PATH=str(runtime.parent)+":"+os.environ.get("LD_LIBRARY_PATH",""),
                 NPU_DYNAMIC_REBASE_INDICES="1",NPU_DYNAMIC_REBASE_FORWARD="1",ACLNN_CACHE_LIMIT="65536")
        subprocess.run([sys.executable,"-B",str(src),str(a.workspace),str(dest),"--flag","NPU_CANN_PENDING_LSE","--switch",str(lane)],env=env,check=True)
        rows.append(json.loads(dest.read_bytes())["records"])
    for arow,brow in zip(*rows):
        for key in ("case","loss","overflow","skip_delta","applied","gradient_sha256","before","after"):
            if arow[key]!=brow[key]:raise ValueError((arow["case"],key))
    (a.out/"overflow-summary.json").write_text(json.dumps(dict(source_sha=original_hash,skip=[r["skip_delta"] for r in rows[0]],
        case=[r["case"] for r in rows[0]],counts=[[r["pending_submissions_while_copying"] for r in rs] for rs in rows],
        exact_gradients_weights_and_optimizer_state=True),indent=2))


if __name__=="__main__":main()
