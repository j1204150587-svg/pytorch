"""Install the opt-in contiguous gradient pack after the 07 guards in a staged workspace."""
from pathlib import Path
import shutil
import sys


def replace(s,old,new):
    if s.count(old)!=1:
        raise ValueError(f"unrecognized dynamic runner anchor: {old!r}")
    return s.replace(old,new)


def stage(workspace):
    partition=workspace/"graphed_hstu.py"
    s=partition.read_text(encoding="utf-8")
    s=replace(s,"import json\n","import json\nimport os\n")
    anchor='    counts = {"partition": 0}\n'
    s=replace(s,anchor,anchor+'''    optimized = None
    if os.getenv("NPU_DYNAMIC_PARTITION_PACK")=="1":
        from partition_pack import GradientPack
        if len(partitions)!=1 or partitions[0][3] is not None or zero.communication_data_type not in (torch.bfloat16, torch.float32):
            raise RuntimeError("gradient pack requires one unpadded unscaled BF16 partition")
        _,shapes,sizes,_ = partitions[0]
        optimized = GradientPack(shapes,sizes,device,zero.dtype)
        zero._eager_gradient_pack = optimized
''')
    a=s.index("            flattened = torch.cat([grad.view(-1)")
    z=s.index("            self._recsys_local_flat_gradients",a)
    original=s[a:z]
    s=s[:a]+'''            packed = None
            if optimized is not None:
                if self.averaged_gradients.get(group_index) is not None:
                    raise RuntimeError("previous optimizer partition has not been cleared")
                packed = optimized.pack(gradients)
            if packed is not None:
                views,flattened = packed
            else:
'''+ "".join("    "+line for line in original.splitlines(keepends=True))+s[z:]
    compile(s,str(partition),"exec")
    partition.write_text(s,encoding="utf-8")
    train=workspace/"train_deepspeed.py"
    s=train.read_text(encoding="utf-8")
    anchor="                    engine.step()\n"
    s=replace(s,anchor,'''                    if pack is not None:
                        pack.before_step()
'''+anchor)
    anchor="    metrics = []\n"
    s=replace(s,anchor,'''    pack = getattr(engine.optimizer, "_eager_gradient_pack", None)
'''+anchor)
    anchor='        end = time.perf_counter()\n'
    s=replace(s,anchor,anchor+'''        pack_stats = dict(enabled=pack is not None, calls=0, hits=0, fallbacks=0, cat_calls=0)
        if pack is not None:
            pack_stats.update(pack.counts)
        (rank_dir / "eager_partition_pack.json").write_text(json.dumps(pack_stats, indent=2))
''')
    compile(s,str(train),"exec")
    train.write_text(s,encoding="utf-8")
    shutil.copyfile(Path(__file__).with_name("partition_pack.py"),workspace/"partition_pack.py")


if __name__=="__main__":
    stage(Path(sys.argv[1]))
