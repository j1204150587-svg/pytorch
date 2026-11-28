"""Replace only SideSlab.forward's live per-table checks with a selectable host guard."""
from pathlib import Path
import shutil
import sys


def stage(workspace):
    path = workspace / "side_slab.py"
    src = path.read_text(encoding="utf-8")
    checks = """        if (len(m.side_embeddings) != len(self.tables) or any(
                current is not prior or current.weight is not w or w.data_ptr() != ptr or not w.requires_grad
                or w.grad is not None or has_hooks(current,w)
                for current,prior,w,ptr in zip(m.side_embeddings,self.tables,self.weights,self.pointers))):
"""
    init = "        self.tables = tuple(model.side_embeddings)\n"
    fn = "def global_hooks():\n"
    for anchor in (checks, init, fn):
        if src.count(anchor) != 1:
            raise ValueError(f"unexpected side-slab implementation at {anchor!r}")
    expr = (checks.strip().removeprefix("if ").removesuffix(":")
            .replace("m.side_embeddings", "live").replace("self.", ""))
    src = src.replace(fn, "def python_parameter_guard(live,tables,weights,pointers):\n    return not "
                      + expr + "\n\n\n" + fn)
    src = src.replace(init, init + '        from hstu_slab_guard import SlabParameterGuard\n'
                      '        self.parameter_guard = SlabParameterGuard(stats)\n')
    src = src.replace(checks, '        if not self.parameter_guard.matches(self,m.side_embeddings,python_parameter_guard):\n')
    compile(src, str(path), "exec")
    path.write_text(src, encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("adapter.py"), workspace / "hstu_slab_guard.py")


if __name__ == "__main__":
    stage(Path(sys.argv[1]))
