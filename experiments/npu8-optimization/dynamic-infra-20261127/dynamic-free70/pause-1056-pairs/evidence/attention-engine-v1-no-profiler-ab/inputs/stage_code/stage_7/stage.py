"""Stage the opt-in first-order eager attention adapter on a training workspace."""
from pathlib import Path
import sys
import textwrap


def unique(src, key, value):
    if src.count(key) != 1:
        raise ValueError("Unexpected target source near " + key[:90])
    return src.replace(key, value)


def stage(workspace, with_runner=True):
    model = workspace / "model.py"
    train = workspace / "train_deepspeed.py"
    src = model.read_text(encoding="utf8")
    if "import os\n" not in src:
        src = unique(src, "import math\n", "import math\nimport os\n")
    anchor = "import torch.nn.functional as F\n"
    src = unique(src, anchor, anchor + 'torch.ops.load_library(os.environ["NPU_ATTENTION_ENGINE_LIBRARY"])\n')
    anchor = "        self.attn_alpha = 1.0 / math.sqrt(attention_dim)\n"
    src = unique(src, anchor, anchor + '''        self._attn_engine = os.getenv("NPU_DYNAMIC_ATTENTION_ENGINE") == "1"
        self._attn_engine_calls = self._attn_engine_hits = 0
''')
    first = "        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)\n"
    last = "        attn = torch.matmul(scores, v).transpose(1, 2).reshape(batch_size, seq_len, -1)\n"
    if src.count(first) != 1 or src.count(last) != 1:
        raise ValueError("Unexpected attention core")
    start = src.index(first)
    end = src.index(last, start) + len(last)
    original = src[start:end]
    if original != '''        q = q.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.attention_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.hidden_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = F.silu(scores).masked_fill(~causal, 0.0)
        scores = scores / float(seq_len)
        attn = torch.matmul(scores, v).transpose(1, 2).reshape(batch_size, seq_len, -1)
''':
        raise ValueError("Attention formulas were modified by another stage")
    wrapper = '''        self._attn_engine_calls += 1
        if (self._attn_engine and self.training and torch.is_grad_enabled()
                and self.num_heads == 4 and self.hidden_dim == self.attention_dim == 16
                and self.attn_alpha == 0.25 and type(uvqk) is torch.Tensor
                and uvqk.dtype == torch.bfloat16 and uvqk.device.type in ("cpu", "npu")
                and uvqk.requires_grad and uvqk.is_contiguous() and uvqk.layout == torch.strided):
            self._attn_engine_hits += 1
            attn = torch.ops.npu_hstu_attn_engine.attention.default(v, q, k)
        else:
'''
    src = src[:start] + wrapper + textwrap.indent(original, "    ") + src[end:]
    compile(src, str(model), "exec")
    if not with_runner:
        model.write_text(src, encoding="utf8")
        return
    entry = train.read_text(encoding="utf8")
    anchor = "        end = time.perf_counter()\n"
    entry = unique(entry, anchor, anchor + '''        attn_enabled = os.getenv("NPU_DYNAMIC_ATTENTION_ENGINE") == "1"
        attn_rows = [{"calls": z._attn_engine_calls, "hits": z._attn_engine_hits,
                      "fallbacks": z._attn_engine_calls - z._attn_engine_hits if attn_enabled else 0}
                     for z in model.blocks]
        attn_ok = len(attn_rows) == 4 and all(r["calls"] == args.steps and r["hits"] == (
            args.steps if attn_enabled else 0) and r["fallbacks"] == 0 for r in attn_rows)
        (rank_dir / "attn_engine_counts.json").write_text(json.dumps(attn_rows, indent=2))
        (rank_dir / "attn_engine_total.json").write_text(json.dumps({"enabled": attn_enabled, "all_expected": attn_ok}))
''')
    compile(entry, str(train), "exec")
    model.write_text(src, encoding="utf8")
    train.write_text(entry, encoding="utf8")


if __name__ == "__main__":
    stage(Path(sys.argv[1]))
