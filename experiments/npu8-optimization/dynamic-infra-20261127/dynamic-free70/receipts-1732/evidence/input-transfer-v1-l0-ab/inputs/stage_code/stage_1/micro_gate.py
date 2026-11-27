"""Verify gated ZeRO flat layout, eager slab gradients, fallback, and partition injection."""
import argparse
from importlib.util import module_from_spec, spec_from_file_location
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace

import torch


def stage_module(source, name):
    spec = spec_from_file_location(name, source/"stage.py")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_engine(model):
    # Mirrors the BF16 dense storage and 07 leaf-consumption boundary for the
    # isolated oracle. Overflow/AdamW in DeepSpeed are covered by the real gate.
    ps = list(model.parameters())
    flat = torch.cat([p.detach().reshape(-1) for p in ps])
    with torch.no_grad():
        offset = 0
        for p in ps:
            p.data = flat.narrow(0,offset,p.numel()).view(p.shape)
            offset += p.numel()
    zero = SimpleNamespace(params_in_partition=[ps],groups_padding=[0],first_offset=[0],
                           bit16_groups_flat=[flat],_grad_acc_hooks=[],_recsys_local_flat_gradients={})
    def partition():
        if any(p.grad is None for p in ps):
            raise ValueError("missing parameter gradient")
        zero.captured = torch.cat([p.grad.detach().reshape(-1) for p in ps])
        for p in ps:
            p.grad = None
    zero.overlapping_partition_gradients_reduce_epilogue = partition
    return SimpleNamespace(module=model,optimizer=zero,world_size=1,gradient_accumulation_steps=lambda:1)


def run(workspace, root, device):
    for name in ("model.py","train_deepspeed.py"):
        shutil.copyfile(workspace/name,root/name)
    here = Path(__file__).resolve().parent
    stage_module(here.parent/"side_tables","group_stage").stage(root)
    stage_module(here,"slab_stage").stage(root)
    sys.path.insert(0,str(root))
    from model import MultiTableHSTU,ModelConfig
    from side_slab import install
    cfg = ModelConfig(**json.loads((workspace/"configs/model_many_tables.json").read_bytes()))
    engines=[]
    for switch in ("0","1"):
        torch.manual_seed(1234)
        os.environ["NPU_DYNAMIC_GROUPED_SIDE"]="1"
        os.environ["NPU_DYNAMIC_SIDE_SLAB"]=switch
        model=MultiTableHSTU(cfg).to(device=device,dtype=torch.bfloat16)
        engine=make_engine(model)
        install(engine)
        engines.append(engine)
    a,b=engines
    if b.module._side_slab_stats["installed"]!=1:
        raise RuntimeError(f"not activated: {b.module._side_slab_stats}")
    rng=torch.Generator().manual_seed(1234)
    rows=[]
    for step,(batch,length) in enumerate(((1,19),(2,14),(2,22),(4,31))):
        tokens=torch.randint(1,cfg.vocab_size,(batch,length+1),generator=rng).to(device)
        ids=torch.randint(cfg.table_rows,(batch,length,cfg.num_tables,4),generator=rng)
        ids[...,1]=ids[...,0]
        ids=ids.to(device)
        groups=torch.randint(cfg.num_user_groups,(batch,),generator=rng).to(device)
        if step==3:
            # A user hook must see the original BF16 grad on the original leaf.
            observed = []
            handles=[e.module.side_embeddings[1].weight.register_hook(lambda gradient:observed.append(gradient.shape))
                     for e in engines]
        logits=[]
        for e in engines:
            out=e.module(tokens[:,:-1].contiguous(),groups,ids)
            torch.nn.functional.cross_entropy(out.float().reshape(-1,cfg.vocab_size),tokens[:,1:].reshape(-1)).backward()
            logits.append(out.detach())
            e.optimizer.overlapping_partition_gradients_reduce_epilogue()
        assert torch.equal(logits[0],logits[1])
        ga,gb=[e.optimizer.captured for e in engines]
        assert torch.equal(ga,gb)
        with torch.no_grad():
            for e in engines:
                if step==1:
                    e.optimizer.captured[-1]=float("inf")
                    assert not bool(torch.isfinite(e.optimizer.captured).all())
                else:
                    e.optimizer.bit16_groups_flat[0].add_(e.optimizer.captured,alpha=-.01)
        assert torch.equal(a.optimizer.bit16_groups_flat[0],b.optimizer.bit16_groups_flat[0])
        rows.append(dict(step=step,shape=[batch,length],equal_logits=True,equal_gradient_elements=ga.numel(),
                         model_parameters=len(list(b.module.parameters())),updated=step!=1))
    assert len(observed)==2
    for handle in handles:handle.remove()
    stats=dict(b.module._side_slab_stats)
    assert stats["runtime_fallback_parameter"]==1 and stats["dense_gradients"]==stats["slab_forwards"]==3
    assert stats["published_table_views"]==3072
    return dict(passed=True,cases=rows,slab_counts=stats,device=device)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace",type=Path)
    parser.add_argument("out",type=Path)
    parser.add_argument("--device",choices=("cpu","npu"),default="cpu")
    args=parser.parse_args()
    if args.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    else:torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as root:
        result=run(args.workspace,Path(root),args.device)
    args.out.write_text(json.dumps(result,indent=2)+"\n")
