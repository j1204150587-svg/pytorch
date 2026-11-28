"""Apply the side-ID route switch to a materialized native RMS eager workspace."""
import json
from pathlib import Path
import shutil


def one(src, old, new):
    if src.count(old)!=1:
        raise ValueError("unexpected staged HSTU source for "+old.splitlines()[0])
    return src.replace(old,new)


def stage(workspace):
    source=Path(__file__).parent
    operators=workspace/"side_tables_op.py"
    train=workspace/"train_deepspeed.py"
    previous=operators.read_text(encoding="utf8")
    old="""    local_ids = ids.movedim(2, 0)
    global_ids = torch.where((local_ids >= 0) & (local_ids < rows),
                             local_ids + offsets, tables * rows).contiguous()
"""
    new="""    with torch.autograd.profiler.record_function("HSTUSideIDs"):
        if weight.ndim == 2 and weight.shape == (65536,64) and weight.dtype == torch.bfloat16:
            global_ids = prepare_ids(ids, offsets, rows, SIDE_IDS_ROUTE)
        else:
            global_ids = prepare_ids(ids, offsets, rows, "A")
"""
    code=one(previous,"import torch.nn.functional as F\n",
             "import torch.nn.functional as F\nimport os\nfrom side_ids_impl import prepare_ids\n"
             "SIDE_IDS_ROUTE = os.getenv('NPU_DYNAMIC_SIDE_IDS_ROUTE', 'A')\n")
    code=one(code,old,new)
    ts=train.read_text(encoding="utf8")
    anchor="        end = time.perf_counter()\n"
    ts=one(ts,anchor,anchor+"""        from side_ids_impl import receipt as side_ids_receipt
        side_ids_receipt(rank_dir/"side_ids.json", args.steps)
""")
    compile(code,str(operators),"exec");compile(ts,str(train),"exec")
    operators.write_text(code,encoding="utf8")
    train.write_text(ts,encoding="utf8")
    for name in ("side_ids_triton.py","side_ids_impl.py","side_ids_driver.py"):
        shutil.copyfile(source/name,workspace/name)


if __name__=="__main__":
    import sys
    stage(Path(sys.argv[1]))
