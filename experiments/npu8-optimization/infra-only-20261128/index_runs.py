"""Validate original runner receipts and index the raw artifacts from each native pair."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

def read(p):
    return json.loads(p.read_bytes())

def dig(p):
    with p.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def manifest(p):
    return {name:sha for line in p.read_text(encoding="utf-8-sig").splitlines() if line
            for sha,name in [line.split("  ",1)]}

def visit(study, pair):
    e=read(pair/"result.json")
    if e["exit"]!=0:
        raise ValueError(pair)
    raw_id=PurePosixPath(e["directory"]).name
    raw=study/"evidence/raw"/raw_id
    model=manifest(raw/"workspace.sha256")
    ref=read(study/"runner/reference.json")
    assert model==ref["workspace"],str(pair)
    cfg=read(pair/"recipe.json")
    schedule=read(study/"evidence/archived-runner/dynamic_shapes_npu8.json")
    legs=[]
    for p in sorted(pair.glob("0?-?")):
        source=raw/p.name
        dispatch=read(p/"dispatch.json")
        leg=p.name[-1]
        argv=dispatch["actual"]
        env=dict(argv[i+1].split("=",1) for i,s in enumerate(argv) if s=="--env")
        assert dispatch["workspace"]==model and env["TASK_QUEUE_ENABLE"]=="1"
        assert {k:v for k,v in env.items() if k.startswith("RECSYS_")} in (
            {"RECSYS_ZERO2_BATCH_OVERFLOW":"1"},{"RECSYS_ZERO2_BATCH_OVERFLOW":"1","RECSYS_DYNAMIC_STATE_AUDIT":"1"})
        assert env[cfg["switch"]]==cfg[leg] and dispatch["libraries"]==cfg["libraries"]
        assert argv[argv.index("--shape-schedule")+1]=="configs/dynamic_shapes_npu8.json"
        assert argv[argv.index("--seed")+1]=="1234"
        data=read(source/"rank000/metrics.json")
        actual=[[r["batch_size"],r["seq_len"]] for r in data["steps"]]
        assert actual==schedule[:len(actual)]
        worker=read(p/"worker.json")
        assert worker["libraries"].keys()==cfg["libraries"].keys()
        files=manifest(source/"evidence.sha256")
        keep={"rank000/metrics.json","rank000/state_audit.json","step_trace_time.csv","trace_view.json.gz"}
        if e["mode"] in ("l0","l1"):
            assert "trace_view.json.gz" in files and "step_trace_time.csv" in files
        items=[]
        for name,expected in files.items():
            if name not in keep and not name.endswith("/kernel_details.csv"): continue
            fp=source/name
            if os.name=="nt":
                fp=Path("//?/"+fp.resolve().as_posix())
            assert dig(fp)==expected,str(fp)
            items.append(dict(file=name,sha256=expected,bytes=fp.stat().st_size))
        legs.append(dict(leg=p.name,worker_host_pid=worker["pid"],actual_shapes=len(actual),
                         effective_argv=f"evidence/pairs/{pair.name}/{p.name}/dispatch.json",files=items))
    assert len(legs)==2
    return dict(name=pair.name,mode=e["mode"],audit=e["audit"],order=e["order"],host_raw=e["directory"],
                local_raw="evidence/raw/"+raw_id,workspace_manifest_sha256=dig(raw/"workspace.sha256"),legs=legs)

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("study",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    cases=[visit(args.study,x) for x in sorted((args.study/"evidence/pairs").iterdir()) if (x/"result.json").is_file()]
    args.out.write_text(json.dumps(cases,indent=2)+"\n",encoding="utf-8")
    print(f"validated {len(cases)} paired sessions and {2*len(cases)} legs")

