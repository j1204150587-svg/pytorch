"""Use a prebuilt native RMS library on both A and B; only B calls the operator."""
from pathlib import Path
import sys


def stage(ws):
    path=ws/"model.py"
    src=path.read_text(encoding="utf8")
    if "import os\n" not in src:
        src=src.replace("import math\n","import math\nimport os\n",1)
    anchor="import torch.nn.functional as F\n"
    init="        self.eps = eps\n"
    forward="        variance = x.pow(2).mean(dim=-1, keepdim=True)\n"
    if any(src.count(s)!=1 for s in (anchor,init,forward)):
        raise ValueError("unexpected HSTU RMS definition")
    src=src.replace(anchor,anchor+'\ntorch.ops.load_library(os.environ["NPU_NATIVE_RMS_LIBRARY"])\n')
    src=src.replace(init,init+'        self._native_rms = os.getenv("NPU_DYNAMIC_NATIVE_RMS") == "1"\n'
                    +'        self._rms_calls = self._rms_hits = 0\n')
    src=src.replace(forward,'''        self._rms_calls += 1
        if (self._native_rms and x.ndim == 3 and x.shape[-1] == 64 and self.weight.shape == (64,)
                and x.dtype == self.weight.dtype == torch.bfloat16 and x.device == self.weight.device
                and self.training and torch.is_grad_enabled()):
            self._rms_hits += 1
            return torch.ops.npu_hstu_native_rms.rms.default(x, self.weight, self.eps)
'''+forward)
    train=ws/"train_deepspeed.py"
    ts=train.read_text(encoding="utf8")
    anchor="        end = time.perf_counter()\n"
    if ts.count(anchor)!=1:
        raise ValueError("unexpected training output")
    ts=ts.replace(anchor,anchor+'''        rms_rows = [{
            "name": name, "enabled": z._native_rms, "calls": z._rms_calls, "hits": z._rms_hits,
            "fallbacks": z._rms_calls - z._rms_hits if z._native_rms else 0
        } for name,z in model.named_modules() if hasattr(z,"_rms_calls")]
        (rank_dir / "native_rms_counts.json").write_text(json.dumps(rms_rows, indent=2))
        rms_enabled = os.getenv("NPU_DYNAMIC_NATIVE_RMS") == "1"
        rms_ok = (len(rms_rows) == 9 and all(r["enabled"] == rms_enabled and r["calls"] == args.steps
                  and r["hits"] == (args.steps if rms_enabled else 0) and r["fallbacks"] == 0
                  for r in rms_rows))
        (rank_dir / "native_rms_total.json").write_text(json.dumps({"enabled": rms_enabled, "all_expected": rms_ok}))
''')
    compile(src,str(path),"exec")
    compile(ts,str(train),"exec")
    path.write_text(src,encoding="utf8")
    train.write_text(ts,encoding="utf8")


if __name__=="__main__":
    stage(Path(sys.argv[1]))
