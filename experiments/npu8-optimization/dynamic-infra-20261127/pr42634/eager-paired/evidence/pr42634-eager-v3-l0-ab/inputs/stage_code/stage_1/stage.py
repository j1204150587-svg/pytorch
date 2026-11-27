"""Layer the side-slab candidate over a workspace already staged for grouped_side_sum."""
from pathlib import Path
import shutil


def replace(text, source, target):
    if text.count(source) != 1:
        raise ValueError(f"unexpected workspace at {source!r}")
    return text.replace(source,target)


def stage(workspace):
    op = workspace/"side_tables_op.py"
    code = op.read_text(encoding="utf8")
    code = replace(code, "    weight = torch.cat(weights, dim=0)\n",
                   "    return sum_from_weight(torch.cat(weights, dim=0), ids, offsets, rows)\n\n\n"
                   "def sum_from_weight(weight, ids, offsets, rows):\n    tables = offsets.shape[0]\n")
    anchor = '_library.impl("grouped_side_sum", grouped_side_sum, "CompositeImplicitAutograd")\n'
    code = replace(code,anchor,anchor +
                   '_library.define("slab_side_sum(Tensor weight, Tensor ids, Tensor offsets, int rows) -> Tensor")\n'
                   '_library.impl("slab_side_sum", sum_from_weight, "CompositeImplicitAutograd")\n')
    anchor = "    model._side_calls += 1\n"
    code = replace(code,anchor,anchor+"""    owner = getattr(model, "_side_slab_owner", None)
    if owner is not None:
        result = owner.forward(ids)
        if result is not None:
            model._side_hits += 1
            return result
""")
    train = workspace/"train_deepspeed.py"
    ts = train.read_text(encoding="utf8")
    # Install after 07 and any optimizer pack wrappers; all derivative publication
    # happens immediately before the existing 07 epilogue.
    anchor = "    if audit:\n        import hashlib\n"
    ts = replace(ts,anchor,"    from side_slab import install as install_side_slab\n    install_side_slab(engine)\n"+anchor)
    anchor = "        end = time.perf_counter()\n"
    ts = replace(ts,anchor,anchor+"        from side_slab import finish as finish_side_slab\n        finish_side_slab(model, rank_dir)\n")
    compile(code,str(op),"exec")
    compile(ts,str(train),"exec")
    op.write_text(code,encoding="utf8")
    train.write_text(ts,encoding="utf8")
    shutil.copyfile(Path(__file__).with_name("side_slab.py"),workspace/"side_slab.py")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace",type=Path)
    stage(parser.parse_args().workspace)
