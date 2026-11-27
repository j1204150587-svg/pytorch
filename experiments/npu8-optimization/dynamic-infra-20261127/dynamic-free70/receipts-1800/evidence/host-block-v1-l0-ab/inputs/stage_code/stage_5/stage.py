"""Install after grouped side, slab, partition pack, and optional flat-statistics overlays."""
from pathlib import Path
import shutil
import sys


def replace(text,old,new):
    if text.count(old)!=1:
        raise ValueError(f"unexpected staged source: {old!r}")
    return text.replace(old,new)


def edit(path,old,new):
    text=replace(path.read_text(encoding="utf8"),old,new)
    compile(text,str(path),"exec")
    path.write_text(text,encoding="utf8")


def stage(workspace):
    anchor='        with torch.autograd.profiler.record_function("EagerPartitionPackCat"):\n'
    edit(workspace/"partition_pack.py",anchor,
         "        return self.commit_pieces(pieces,merged)\n\n    def commit_pieces(self,pieces,merged):\n"+anchor)
    anchor="        state.publish()\n        return original(*args, **kwargs)\n"
    edit(workspace/"side_slab.py",anchor,
         '        direct=getattr(self,"_direct_side_publisher",None)\n'
         '        if direct is None or not direct.prepare():\n'
         '            state.publish()\n        return original(*args, **kwargs)\n')
    anchor="            gradients = [parameter.grad for parameter in group]\n"
    edit(workspace/"graphed_hstu.py",anchor,
         '''            direct=getattr(self,"_direct_side_publisher",None)
            if direct is not None and direct.ready is not None:
                if group_index!=0 or group is not direct.state.group:
                    raise RuntimeError("side gradient delivered to wrong partition")
                views,flattened=direct.commit()
                self._recsys_local_flat_gradients[group_index]=(views,flattened)
                self.averaged_gradients[group_index]=views
                self.all_grad_tensors[group_index]=None
                self.free_grad_in_param_list(direct.ordinary)
                continue
'''+anchor)
    anchor="    install_side_slab(engine)\n"
    edit(workspace/"train_deepspeed.py",anchor,anchor+"    from side_publish import install as install_direct_side\n    install_direct_side(engine)\n")
    flat_stats=workspace/"flat_stats.py"
    if flat_stats.is_file():
        anchor="        # Validate actual layout, including replacement/reordering inside the\n"
        edit(flat_stats,anchor,
             '        direct=getattr(zero,"_direct_side_publisher",None)\n'
             '        if direct is not None and direct.owns(shards,flat):\n'
             '            direct.stats["trusted_stats_reads"]+=1\n'
             '            return flat\n'+anchor)
    shutil.copyfile(Path(__file__).with_name("side_publish.py"),workspace/"side_publish.py")


if __name__=="__main__":
    stage(Path(sys.argv[1]))
