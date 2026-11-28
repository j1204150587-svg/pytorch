"""Install the same score ELF on both legs; switch only the host submitter."""
from pathlib import Path
import shutil
import sys


def replace(src,anchor,replacement):
    if src.count(anchor)!=1:
        raise ValueError("unexpected score stage source near "+anchor[:70])
    return src.replace(anchor,replacement)


def stage(ws):
    here=Path(__file__).resolve().parent
    model=ws/"model.py"
    train=ws/"train_deepspeed.py"
    m=model.read_text(encoding="utf8")
    if "import os\n" not in m:
        m=replace(m,"import math\n","import math\nimport os\n")
    mark="import torch.nn.functional as F\n"
    m=replace(m,mark,mark+"import hstu_masked_score as _score\n")
    mark="        self.attn_alpha = 1.0 / math.sqrt(attention_dim)\n"
    m=replace(m,mark,mark+'        self._score_planned = os.getenv("NPU_DYNAMIC_LOCKED_SCORE_PLAN") == "1"\n')
    original='''        scores = torch.matmul(q, k.transpose(-2, -1)) * self.attn_alpha
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        scores = F.silu(scores).masked_fill(~causal, 0.0)
        scores = scores / float(seq_len)
'''
    fallback='''            scores = raw_scores * self.attn_alpha
            causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
            scores = F.silu(scores).masked_fill(~causal, 0.0) / float(seq_len)
'''
    m=replace(m,original,'''        raw_scores = torch.matmul(q, k.transpose(-2,-1))
        if self.attn_alpha == 0.25 and self.training and torch.is_grad_enabled():
            scores = _score.apply(raw_scores,self._score_planned)
        else:
'''+fallback)
    t=train.read_text(encoding="utf8")
    mark="        end = time.perf_counter()\n"
    t=replace(t,mark,mark+'''        import hstu_score_lock as score_lock
        import hstu_masked_score as score_impl
        score_enabled = os.getenv("NPU_DYNAMIC_LOCKED_SCORE_PLAN") == "1"
        score_counts = dict(score_lock.COUNTS)
        score_services = score_lock.service_counts()
        score_expected = 4 * args.steps
        score_ok = not score_counts.get("fallback") and not score_impl.COUNTS.get("layout_fallback")
        for score_name in ("causal_score_forward", "causal_score_backward"):
            score_cold = 1 if score_enabled else score_expected
            score_plans = score_expected - score_cold
            score_ok = score_ok and score_counts.get("jit_"+score_name, 0) == score_cold
            score_ok = score_ok and score_counts.get("planned_"+score_name, 0) == score_plans
            score_ok = score_ok and score_services.get(score_name, [0,0,0]) == [score_plans]*3
        score_data = {"enabled": score_enabled, "all_expected": score_ok, "calls": score_counts,
            "service_counts": score_services, "plans": score_lock.EVIDENCE, "impl": dict(score_impl.COUNTS)}
        (rank_dir / "locked_score_counts.json").write_text(json.dumps(score_data,indent=2))
''')
    if not (ws/"fold_probe.py").is_file():raise ValueError("missing common fold-probe ABI helper")
    impl=(here/"operator_impl.py").read_text(encoding="utf8")
    impl=replace(impl,"from kernels import ","from hstu_masked_score_kernels import ")
    impl+="\nimport hstu_score_lock as _lock\nlaunch = _lock.launch\n"
    for code,name in ((m,model),(t,train),(impl,ws/"hstu_masked_score.py")):
        compile(code,str(name),"exec")
        name.write_text(code,encoding="utf8")
    shutil.copyfile(here/"locked.py",ws/"hstu_score_lock.py")
    shutil.copyfile(here/"kernels.py",ws/"hstu_masked_score_kernels.py")


if __name__=="__main__":
    stage(Path(sys.argv[1]))
