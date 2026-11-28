"""Insert the one-switch metadata guard after flat_stats has been installed."""
from pathlib import Path
import shutil
import sys


def stage(workspace):
    path=workspace/"flat_stats.py"
    src=path.read_text(encoding="utf-8")
    start='    def eligible(shards):\n'
    body='        # Validate actual layout, including replacement/reordering inside the\n'
    tail='        return flat\n'
    if src.count(start)!=1 or src.count(body)!=1 or src.count(tail)!=1:
        raise ValueError("unexpected flat_stats eligible implementation")
    begin=src.index(start)
    loop=src.index(body,begin)
    end=src.index(tail,loop)+len(tail)
    original=src[loop:end]
    anchor="        offset = flat.storage_offset()\n"
    check="type(grad) is not torch.Tensor or grad._base is not root or"
    if original.count(anchor)!=1 or original.count(check)!=1:
        raise ValueError("unexpected source storage predicate")
    original=original.replace(anchor,anchor+"        flat_storage = flat.untyped_storage()._cdata\n").replace(
        check,check+" grad.untyped_storage()._cdata != flat_storage or")
    src=src[:begin]+'''    from hstu_metadata_guard import MetadataGuard
    metadata_guard = MetadataGuard(shapes,sizes,rank_dir)

    def original_layout(flat,shards):
'''+original+'\n'+src[begin:loop]+'''        return metadata_guard.evaluate(flat,shards,original_layout)
'''+src[end:]
    compile(src,str(path),"exec")
    path.write_text(src,encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("adapter.py"),workspace/"hstu_metadata_guard.py")


if __name__=="__main__":
    stage(Path(sys.argv[1]))
