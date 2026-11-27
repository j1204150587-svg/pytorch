"""Patch both training legs to share one eager BF16 RMS wrapper and exactly one launch switch."""
from pathlib import Path
import shutil
import sys


def once(text,old,new):
    if text.count(old)!=1:
        raise ValueError(f"unexpected dynamic training source near {old!r}")
    return text.replace(old,new)


def stage(ws):
    here=Path(__file__).resolve().parent
    for name in ("planned_rms.py","lut_rms.py","lut_kernel.py","forward_probe.py"):
        shutil.copyfile(here/name,ws/name)
    # The shared plan verifier is also used by the ordered-fold recipe.
    probe=ws/"fold_probe.py"
    if not probe.is_file():
        local=here/"fold_probe.py"
        if not local.is_file():
            local=here.parent/"pr42634_eager_fold"/"fold_probe.py"
        shutil.copyfile(local,probe)
    model=ws/"model.py"
    source=model.read_text(encoding="utf8")
    if "import os\n" not in source:
        source=once(source,"import math\n","import math\nimport os\n")
    imp="import torch.nn.functional as F\n"
    source=once(source,imp,imp+"\nimport planned_rms as _planned_rms\n"
                    +'_planned_rms.install(os.environ["NPU_RMS_LUT_PARTS_LIBRARY"])\n')
    anchor="        variance = x.pow(2).mean(dim=-1, keepdim=True)\n"
    source=once(source,anchor,"""        if (self.training and torch.is_grad_enabled() and x.ndim == 3 and x.shape[-1] == 64
                and self.weight.shape == (64,) and x.dtype == self.weight.dtype == torch.bfloat16
                and x.device == self.weight.device and x.device.type == "npu"):
            return _planned_rms.run(x,self.weight,self.eps)
"""+anchor)
    entry=ws/"train_deepspeed.py"
    train=entry.read_text(encoding="utf8")
    anchor="        end = time.perf_counter()\n"
    train=once(train,anchor,anchor+'        from planned_rms import finish as _finish_rms_lut\n'
                  +'        _finish_rms_lut(rank_dir/"planned_rms_lut.json",args.steps)\n')
    compile(source,str(model),"exec")
    compile(train,str(entry),"exec")
    model.write_text(source,encoding="utf8")
    entry.write_text(train,encoding="utf8")


if __name__=="__main__":
    stage(Path(sys.argv[1]))
