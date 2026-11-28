"""Check wide FP64 conversion and adjacent row-norm batches on the model's real layout."""
import argparse
import json
from pathlib import Path
import sys
import types
import torch
from wide_norm import join_adjacent_batches


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--workspace",type=Path,required=True)
    cli.add_argument("--device",choices=("cpu","npu"),required=True)
    cli.add_argument("--output",type=Path,required=True)
    a=cli.parse_args()
    sys.path.insert(0,str(a.workspace.resolve()))
    from model import MultiTableHSTU,ModelConfig
    from flat_stats import norm_plan,row_norm_square
    if a.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    torch.set_num_threads(1)
    with torch.device("meta"):
        model=MultiTableHSTU(ModelConfig(num_tables=1024,table_rows=64))
    params=list(model.parameters())
    stage=types.SimpleNamespace(PIPE_REPLICATED="pipeline_replicated",is_model_parallel_parameter=lambda p:False)
    plan=norm_plan(params,stage,types.SimpleNamespace(model_parallel_rank=0),256*1024,64)
    count=sum(p.numel() for p in params)
    join=join_adjacent_batches(plan)
    seed=torch.Generator().manual_seed(1234)
    rows=[]
    for tag in ("standard","wide","near_clip","zero","inf","nan"):
        flat=torch.randn(count,generator=seed).bfloat16()
        if tag=="wide": flat[::181]*=float(2**15)
        elif tag=="near_clip": flat/=count**.5
        elif tag=="zero": flat.zero_()
        elif tag=="inf": flat[4095]=float("inf")
        elif tag=="nan": flat[-1]=float("nan")
        flat=flat.to(a.device)
        ref=row_norm_square(flat,plan).cpu()
        whole=flat.double()
        cast=row_norm_square(whole,plan).cpu()
        merged=row_norm_square(whole,join).cpu()
        equal=lambda x:torch.equal(ref.view(torch.int32),x.view(torch.int32)) or bool(torch.isnan(ref) and torch.isnan(x))
        rows.append(dict(tag=tag,cast_exact=equal(cast),join_exact=equal(merged)))
    ok=all(r['cast_exact'] and r['join_exact'] for r in rows)
    a.output.write_text(json.dumps(dict(device=a.device, torch=torch.__version__, params=len(params), elements=count,
                                        groups=len(plan), joined_groups=len(join), cases=rows, passed=ok),indent=2)+"\n")
    raise SystemExit(0 if ok else 1)
