"""Enable the archived rank-one flat/weight hints after dynamic eager partition installation."""
from pathlib import Path
import shutil
import sys


def stage(workspace):
    train=workspace/"train_deepspeed.py"
    s=train.read_text(encoding="utf-8")
    anchor="        norm_stats = install_norm(engine.optimizer)\n"
    if s.count(anchor)!=1:
        raise ValueError("unexpected eager norm installation")
    s=s.replace(anchor,anchor+'''    hint_finish = None
    if os.getenv("NPU_DYNAMIC_ZERO_FLAT_HINTS")=="1":
        from local_step import install as install_zero_hints
        hint_finish = install_zero_hints(engine.optimizer, device, "both", rank_dir)
''')
    anchor='        end = time.perf_counter()\n'
    if s.count(anchor)!=1:
        raise ValueError("unexpected synchronized window")
    s=s.replace(anchor,anchor+'''        if hint_finish is not None:
            hint_finish()
            hint = json.loads((rank_dir / "zero2_local_step.json").read_text())
            hint_semantic_path = rank_dir / "semantic.json"
            hint_updates = sum(r["applied"] for r in json.loads(hint_semantic_path.read_text())) if hint_semantic_path.is_file() else args.steps
            if hint["flatten"] != hint_updates or hint["aliases"] != hint_updates:
                raise RuntimeError(f"local optimizer hints missed an update: {hint}, expected={hint_updates}")
        else:
            hint = {"flatten": 0, "aliases": 0}
        (rank_dir / "zero_flat_hints.json").write_text(json.dumps({
            "enabled": hint_finish is not None, "flatten": hint["flatten"], "aliases": hint["aliases"],
            "validated": True}))
''')
    compile(s,str(train),"exec")
    train.write_text(s,encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("local_step.py"),workspace/"local_step.py")


if __name__=="__main__":
    stage(Path(sys.argv[1]))
