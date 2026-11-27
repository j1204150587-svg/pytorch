"""Opt into exact BF16 tril zeroing at the existing causal-score boundary."""
from pathlib import Path
import sys


def stage(workspace):
    path=workspace/"model.py"
    source=path.read_text(encoding="utf8")
    anchor="        self.attn_alpha = 1.0 / math.sqrt(attention_dim)\n"
    if source.count(anchor)!=1:raise ValueError("unexpected HSTUBlock declaration")
    source=source.replace(anchor,anchor+'        self._bf16_tril = os.environ.get("NPU_DYNAMIC_CAUSAL_TRIL") == "1"\n'
                          +'        self._causal_calls = self._causal_hits = 0\n')
    if "import os\n" not in source:
        source=source.replace("import math\n","import math\nimport os\n",1)
    old="""        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = F.silu(scores).masked_fill(~causal, 0.0)
"""
    if source.count(old)!=1:raise ValueError("unexpected causal mask path")
    source=source.replace(old,'''        self._causal_calls += 1
        if self._bf16_tril and scores.dtype == torch.bfloat16 and scores.ndim == 4 and scores.shape[-2:] == (seq_len,seq_len):
            scores = torch.tril(F.silu(scores))
            self._causal_hits += 1
        else:
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            scores = F.silu(scores).masked_fill(~causal, 0.0)
''')
    train=workspace/"train_deepspeed.py"
    code=train.read_text(encoding="utf8")
    pos="        end = time.perf_counter()\n"
    if code.count(pos)!=1:raise ValueError("unexpected training output")
    code=code.replace(pos,pos+'''        (rank_dir / "causal_tril_counts.json").write_text(json.dumps([{
            "enabled": block._bf16_tril, "calls": block._causal_calls, "hits": block._causal_hits,
            "fallbacks": block._causal_calls - block._causal_hits if block._bf16_tril else 0
        } for block in model.blocks],indent=2))
''')
    compile(source,str(path),"exec")
    compile(code,str(train),"exec")
    path.write_text(source,encoding="utf8")
    train.write_text(code,encoding="utf8")


if __name__=="__main__":
    stage(Path(sys.argv[1]))
