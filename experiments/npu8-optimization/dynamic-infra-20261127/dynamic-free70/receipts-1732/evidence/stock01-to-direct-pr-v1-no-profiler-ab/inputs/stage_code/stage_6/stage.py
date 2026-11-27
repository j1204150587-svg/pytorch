"""Install after the grouped/slab/direct stages on both matched training workspaces."""
from pathlib import Path
import shutil
import sys


def replace(text, old, new):
    if text.count(old) != 1:
        raise ValueError("unexpected training source at: " + old)
    return text.replace(old,new)


def stage(ws):
    here=Path(__file__).resolve().parent
    op=ws/"side_tables_op.py"
    source=replace(op.read_text(encoding="utf8"), "        from npu_ordered_fold import ordered_64\n",
                   "        from eager_fold_plan import launch\n")
    source=replace(source, "        ordered_64[(means.shape[1] * means.shape[2],)](means.contiguous(), result, means.shape[0])\n",
                   "        launch(means.contiguous(), result)\n")
    entry=ws/"train_deepspeed.py"
    train=entry.read_text(encoding="utf8")
    train=replace(train,"    metrics = []\n","    from eager_fold_plan import begin as fold_begin, finish as fold_finish\n    metrics = []\n")
    anchor="            for step, (batch_size, sequence_length) in enumerate(shapes):\n"
    train=replace(train,anchor,anchor+"                fold_begin(step)\n")
    anchor="        end = time.perf_counter()\n"
    train=replace(train,anchor,anchor+'        fold_finish(rank_dir / "eager_fold_plan.json")\n')
    compile(source,str(op),"exec")
    compile(train,str(entry),"exec")
    for name in ("eager_fold_plan.py","fold_probe.py"):
        shutil.copyfile(here/name,ws/name)
    op.write_text(source,encoding="utf8")
    entry.write_text(train,encoding="utf8")


if __name__ == "__main__":
    stage(Path(sys.argv[1]))
