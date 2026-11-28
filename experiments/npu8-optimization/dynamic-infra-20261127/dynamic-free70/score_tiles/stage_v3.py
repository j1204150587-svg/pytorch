"""Compare ordinary and planned submissions of the same causal-score kernels in eager training."""
from pathlib import Path
import shutil
import sys


def change(source, before, after):
    if source.count(before) != 1:
        raise ValueError("unexpected training source near " + before[:75])
    return source.replace(before, after)


def stage(workspace):
    here = Path(__file__).resolve().parent
    model = workspace / "model.py"
    trainer = workspace / "train_deepspeed.py"
    src = model.read_text()
    if "import os\n" not in src:
        src = change(src, "import math\n", "import math\nimport os\n")
    mark = "import torch.nn.functional as F\n"
    src = change(src, mark, mark + "import hstu_tile_scores as _tile_scores\n")
    mark = "        self.attn_alpha = 1.0 / math.sqrt(attention_dim)\n"
    src = change(src, mark, mark + '        self._score_plan = os.getenv("NPU_DYNAMIC_TILE_SCORE_PLAN") == "1"\n')
    before = '''        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = F.silu(scores).masked_fill(~causal, 0.0)
        scores = scores / float(seq_len)
'''
    after = '''        raw_scores = torch.matmul(q, k.transpose(-2, -1))
        if self.attn_alpha == 0.25 and self.training and torch.is_grad_enabled():
            scores = _tile_scores.apply(raw_scores, self._score_plan)
        else:
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            scores = F.silu(raw_scores * self.attn_alpha).masked_fill(~causal, 0.0) / float(seq_len)
'''
    src = change(src,before,after)
    mark = "        end = time.perf_counter()\n"
    t = change(trainer.read_text(), mark, mark + '''        import hstu_tile_scores as tile_scores
        tile_counts = dict(tile_scores.COUNTS)
        tile_on = os.getenv("NPU_DYNAMIC_TILE_SCORE_PLAN") == "1"
        tile_total = 8 * args.steps
        tile_planned = tile_counts.get("direct", 0) + tile_counts.get("capped", 0)
        tile_expected = tile_total - 2 if tile_on else 0
        tile_ok = (tile_planned == tile_expected and tile_counts.get("jit", 0) == tile_total - tile_expected
                   and tile_counts.get("prepared", 0) == 2 * int(tile_on) and tile_counts.get("table_builds") == 1
                   and all(not tile_counts.get(name) for name in ("layout_fallback","resource_fallback")))
        tile_report = dict(enabled=tile_on, passed=tile_ok, steps=args.steps, counts=tile_counts, plans=tile_scores.EVIDENCE)
        (rank_dir / "tile_score_counts.json").write_text(json.dumps(tile_report, indent=2))
''')
    impl = change((here / "operator_impl.py").read_text(),"from kernels import ","from hstu_tile_kernels import ")
    for file,code in ((model,src),(trainer,t),(workspace / "hstu_tile_scores.py",impl)):
        compile(code,str(file),"exec")
        file.write_text(code)
    if not (workspace / "fold_probe.py").is_file():
        raise ValueError("PR tensor ABI witness is absent from the staged workspace")
    shutil.copyfile(here / "kernels.py", workspace / "hstu_tile_kernels.py")


if __name__ == "__main__":
    stage(Path(sys.argv[1]))
