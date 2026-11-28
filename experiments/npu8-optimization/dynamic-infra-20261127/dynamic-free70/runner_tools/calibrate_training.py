"""Collect a diagnostic Level1 run, appending clock witnesses after measured steps 4 and 5."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def main(args):
    out=args.output.parent
    work=out/"workspace"
    shutil.copytree(args.workspace,work,ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    train=work/"train_deepspeed.py"
    source=train.read_text()
    anchor="                metrics.append(metric)\n"
    if source.count(anchor)!=1:
        raise ValueError("unknown training loop")
    # Prompt completion of the device-wide synchronize tightens the upper bound.
    extra='''                if step == 6:
                    with record_function("ClockCalibrationAfterTrainingStep6"):
                        mark = torch.empty(1048576, dtype=torch.float32, device=engine.device)
                        mark.zero_()
                        torch.npu.synchronize()
                        for calibration_repeat in range(16):
                            with record_function("ClockCalibrationWitness"):
                                mark.add_(1)
                                torch.npu.synchronize()
'''
    train.write_text(source.replace(anchor,anchor+extra))
    cmd=["bash","./run_training.sh","--model-config","configs/model_many_tables.json",
         "--deepspeed-config","configs/deepspeed_zero2_bf16_overflow_on.json","--steps","8",
         "--micro-batch-size","2","--seed","1234","--learning-rate","0.001","--weight-decay","0.01",
         "--log-interval","4","--output-dir",str(out/"training"),"--profile","--profile-active","3",
         "--profile-level1","--shape-schedule","configs/dynamic_shapes_npu8.json"]
    data=dict(workspace=str(args.workspace),command=cmd,training_before=hashlib.sha256(source.encode()).hexdigest(),
              training_after=hashlib.sha256(train.read_bytes()).hexdigest(),calibration_step=6)
    result=subprocess.run(cmd,cwd=work)
    data["exit"]=result.returncode
    args.output.write_text(json.dumps(data,indent=2))
    result.check_returncode()


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",required=True,type=Path)
    cli.add_argument("--output",required=True,type=Path)
    main(cli.parse_args())
