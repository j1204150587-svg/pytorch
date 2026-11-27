"""Stage the optional causal-attention experiment in the existing dynamic eager block."""
from pathlib import Path
import shutil
import sys


def stage(workspace):
    path = workspace / "model.py"
    src = path.read_text(encoding="utf-8")
    init = "        self.attn_alpha = 1.0 / math.sqrt(attention_dim)\n"
    start = "        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha\n"
    end = "        attn = torch.matmul(scores, v).transpose(1, 2).reshape(batch_size, seq_len, -1)\n"
    for anchor in (start, end, init, "import math\n"):
        if src.count(anchor) != 1:
            raise ValueError("unexpected attention block at " + anchor.strip())
    i = src.index(start)
    j = src.index(end,i)+len(end)
    old = src[i:j]
    src = src[:i] + '''        from attention_case import Full, MaskDiv, layout
        self._attention_calls += 1
        enabled = self._attention_mode != "off"
        use = (enabled and self.attn_alpha == .25 and q.device.type == "npu" and torch.is_grad_enabled()
               and layout(q,k,v))
        marker = {"full": "HSTUAttentionFull", "partial": "HSTUAttentionMaskDiv"}
        with torch.autograd.profiler.record_function(marker[self._attention_mode] if use else "HSTUAttentionOriginal"):
            if use:
                self._attention_hits += 1
                if self._attention_mode == "full":
                    inner = Full.apply(q,k,v)
                else:
                    scaled = torch.matmul(q,k.transpose(-2,-1)) * self.attn_alpha
                    inner = torch.matmul(MaskDiv.apply(F.silu(scaled)),v)
                attn = inner.transpose(1,2).reshape(batch_size,seq_len,-1)
            else:
''' + "".join("        "+line for line in old.splitlines(keepends=True)) + src[j:]
    src = src.replace(init,init+'''        self._attention_mode = os.environ.get("NPU_DYNAMIC_CAUSAL_ATTENTION","off")
        if self._attention_mode not in ("off","partial","full"):
            raise ValueError(self._attention_mode)
        self._attention_hits = self._attention_calls = 0
''')
    if "import os\n" not in src:
        src = src.replace("import math\n","import math\nimport os\n",1)
    train = workspace / "train_deepspeed.py"
    main = train.read_text(encoding="utf-8")
    anchor = "        end = time.perf_counter()\n"
    if main.count(anchor) != 1:
        raise ValueError("unexpected timer end")
    main = main.replace(anchor,anchor+'''        checks = [{"mode": b._attention_mode, "calls": b._attention_calls, "hits": b._attention_hits} for b in model.blocks]
        (rank_dir / "causal_attention_counts.json").write_text(json.dumps({
            "blocks": checks, "all_blocks_same": len({(b["calls"],b["hits"],b["mode"]) for b in checks}) == 1,
            "num_blocks": len(checks), "mode": checks[0]["mode"], "calls_each": checks[0]["calls"],
            "hits_each": checks[0]["hits"], "fallbacks":sum(b["calls"]-b["hits"] for b in checks if b["mode"]!="off")
        },indent=2))
''')
    compile(src,str(path),"exec")
    compile(main,str(train),"exec")
    path.write_text(src,encoding="utf-8")
    train.write_text(main,encoding="utf-8")
    for name in ("attention_case.py","attn_kernels.py"):
        shutil.copyfile(Path(__file__).with_name(name),workspace/name)


if __name__ == "__main__":
    stage(Path(sys.argv[1]))
