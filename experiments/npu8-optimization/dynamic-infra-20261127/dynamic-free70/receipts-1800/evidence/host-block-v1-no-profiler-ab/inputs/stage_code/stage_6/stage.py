from pathlib import Path
import shutil
import sys


def unique(text,s,with_s):
    if text.count(s)!=1:
        raise ValueError("unexpected HSTU block source at "+s)
    return text.replace(s,with_s)


def stage(ws):
    p=ws/"model.py"
    s=p.read_text(encoding="utf8")
    if "import os\n" not in s:
        s=unique(s,"import math\n","import math\nimport os\n")
    anchor="import torch.nn.functional as F\n"
    s=unique(s,anchor,anchor+'from block_dispatch import eligible as _host_block_ok, run as _host_block_run\n'
             +'torch.ops.load_library(os.environ["NPU_HOST_BLOCK_LIBRARY"])\n')
    anchor="        self.attn_alpha = 1.0 / math.sqrt(attention_dim)\n"
    s=unique(s,anchor,anchor+'        self._native_block = os.getenv("NPU_DYNAMIC_HOST_BLOCK") == "1"\n'
             +'        self._block_calls = self._block_hits = 0\n')
    anchor="        batch_size, seq_len, _ = x.shape\n"
    s=unique(s,anchor,'''        self._block_calls += 1
        if (self._native_block and type(self.input_norm) is RMSNorm and type(self.output_norm) is RMSNorm
                and _host_block_ok(self,x)):
            self._block_hits += 1
            return _host_block_run(self,x)
'''+anchor)
    entry=ws/"train_deepspeed.py"
    ts=entry.read_text(encoding="utf8")
    a="        end = time.perf_counter()\n"
    ts=unique(ts,a,a+'''        block_rows = [{
            "enabled": z._native_block, "calls": z._block_calls, "hits": z._block_hits,
            "fallbacks": z._block_calls - z._block_hits if z._native_block else 0
        } for z in model.blocks]
        (rank_dir / "host_block_counts.json").write_text(json.dumps(block_rows, indent=2))
        block_enabled = os.getenv("NPU_DYNAMIC_HOST_BLOCK") == "1"
        block_ok = len(block_rows) == 4 and all(r["enabled"] == block_enabled and r["calls"] == args.steps
                       and r["hits"] == (args.steps if block_enabled else 0) and r["fallbacks"] == 0 for r in block_rows)
        (rank_dir / "host_block_total.json").write_text(json.dumps({"enabled": block_enabled, "all_expected": block_ok}))
''')
    compile(s,str(p),"exec")
    compile(ts,str(entry),"exec")
    shutil.copyfile(Path(__file__).with_name("block_dispatch.py"),ws/"block_dispatch.py")
    p.write_text(s,encoding="utf8")
    entry.write_text(ts,encoding="utf8")


if __name__=="__main__":
    stage(Path(sys.argv[1]))
