"""Exercise the staged 07 code on CPU with a single-rank ZeRO layout fixture."""
import argparse
from importlib.util import module_from_spec,spec_from_file_location
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import ModuleType,SimpleNamespace

import torch


class Zero:
    def free_grad_in_param_list(self,ordinary):
        self.last_cleared=len(ordinary)
        for p in ordinary:p.grad=None


def install_ds_fixture():
    for name in ("deepspeed","deepspeed.runtime","deepspeed.runtime.zero","deepspeed.runtime.zero.stage_1_and_2"):
        sys.modules[name]=ModuleType(name)
    dist=SimpleNamespace(get_world_size=lambda *args:1)
    sys.modules["deepspeed"].comm=dist
    sys.modules["deepspeed.runtime.zero.stage_1_and_2"].DeepSpeedZeroOptimizer=Zero


def engine_for(model):
    group=list(model.parameters())
    flat=torch.cat([p.detach().reshape(-1) for p in group])
    offset=0
    for param in group:
        param.data=flat.narrow(0,offset,param.numel()).view_as(param)
        offset+=param.numel()
    z=Zero()
    z.__dict__.update(params_in_partition=[group],bit16_groups=[group],bit16_groups_flat=[flat],
        partition_gradients=True,cpu_offload=False,has_moe_layers=False,zenflow=False,model_parallel_group=None,
        sequence_parallel_size=1,reduce_scatter=True,contiguous_gradients=True,check_grad_overflow=True,
        dtype=torch.bfloat16,gradient_accumulation_dtype=torch.bfloat16,use_grad_accum_attribute=False,
        torch_autocast_gradscaler=None,dp_process_group=None,real_dp_process_group=[None],
        optimizer=object.__new__(torch.optim.AdamW),round_robin_gradients=False,
        _grad_acc_hooks=[SimpleNamespace(remove=lambda:None) for _ in group],partition_size=[flat.numel()],
        first_offset=[0],groups_padding=[0],is_gradient_accumulation_boundary=True,_epilogue_ran_this_backward=False,
        communication_data_type=torch.float32,averaged_gradients={},all_grad_tensors={})
    return SimpleNamespace(module=model,optimizer=z,world_size=1,gradient_accumulation_steps=lambda:1)


def stage_code(workspace,dest):
    for n in ("model.py","graphed_hstu.py","train_deepspeed.py"):shutil.copyfile(workspace/n,dest/n)
    base=Path(__file__).resolve().parents[1]
    for entry in ("side_tables/stage.py","side_slab/stage.py","partition_hints/stage.py",
                  "partition_pack/stage.py","optimizer/stage_flat_stats.py","side_publish/stage.py"):
        spec=spec_from_file_location(entry,base/entry)
        mod=module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod,"stage"):
            mod.stage(dest)
        else:
            subprocess.run([sys.executable,"-B",str(base/entry),str(dest)],check=True)


def run(workspace,dest):
    stage_code(workspace,dest)
    sys.path.insert(0,str(dest))
    from model import MultiTableHSTU,ModelConfig
    from graphed_hstu import install_single_rank_partition
    from side_slab import install as install_slab
    from side_publish import install as install_pub
    cfg=ModelConfig(**json.loads((workspace/"configs/model_many_tables.json").read_bytes()))
    engines=[]
    install_ds_fixture()
    for flag in ("0","1"):
        for name in ("NPU_DYNAMIC_GROUPED_SIDE","NPU_DYNAMIC_SIDE_SLAB","NPU_DYNAMIC_PARTITION_PACK"):
            os.environ[name]="1"
        os.environ["NPU_DYNAMIC_DIRECT_SIDE"]=flag
        torch.manual_seed(1234)
        e=engine_for(MultiTableHSTU(cfg).to(dtype=torch.bfloat16))
        e.count=install_single_rank_partition(e,torch.device("cpu"))
        install_slab(e)
        install_pub(e)
        engines.append(e)
    rng=torch.Generator().manual_seed(125)
    rows=[]
    b=engines[1]
    direct=b.optimizer._direct_side_publisher
    for step,(batch,length) in enumerate(((1,19),(2,14),(2,22),(4,31))):
        tokens=torch.randint(1,cfg.vocab_size,(batch,length+1),generator=rng)
        ids=torch.randint(cfg.table_rows,(batch,length,1024,4),generator=rng)
        ids[...,1]=ids[...,0]
        groups=torch.randint(cfg.num_user_groups,(batch,),generator=rng)
        if step==3:direct.views[0]=direct.views[0]  # guarded mutation of private container
        logits=[]
        for e in engines:
            z=e.optimizer
            z._epilogue_ran_this_backward=False
            z.averaged_gradients[0]=None
            out=e.module(tokens[:,:-1].contiguous(),groups,ids)
            logits.append(out.detach())
            torch.nn.functional.cross_entropy(out.float().reshape(-1,cfg.vocab_size),tokens[:,1:].reshape(-1)).backward()
            assert all(t.weight.grad is None for t in e.module.side_embeddings)
            z.overlapping_partition_gradients_reduce_epilogue()
        a=z0=engines[0].optimizer
        z1=b.optimizer
        left,right=a._recsys_local_flat_gradients[0],z1._recsys_local_flat_gradients[0]
        assert torch.equal(logits[0],logits[1]) and torch.equal(left[1],right[1])
        assert len(right[0])==1052 and all(torch.equal(x,y) for x,y in zip(left[0],right[0]))
        assert direct.owns(*right)==(step<3)
        assert z1.last_cleared==(28 if step<3 else 1052)
        for z in (z0,z1):
            views,flat=z._recsys_local_flat_gradients[0]
            z._eager_gradient_pack.before_step()
            if step==1:
                views[7][0,-1]=float("inf")
                assert torch.isinf(flat).any()
            else:
                z.bit16_groups_flat[0].add_(flat,alpha=-.01)
        if step==1:assert not direct.owns(*right)
        assert torch.equal(z0.bit16_groups_flat[0],z1.bit16_groups_flat[0])
        rows.append(dict(step=step,shape=[batch,length],all_1052_gradients_equal=True,
                         weight_equal=True,ordinary_clear_count=z1.last_cleared))
    stats=dict(b.module._side_slab_stats)
    assert stats["direct_partitions"]==3 and stats["ordinary_gradients_checked"]==3*28
    assert stats["direct_final_views"]==3156 and stats["gradient_buffer_copies"]==1
    assert stats["direct_fallback_ownership"]==1 and b.count["partition"]==4
    return dict(passed=True,torch=str(torch.__version__),device="cpu",cases=rows,counts=stats)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace",type=Path)
    parser.add_argument("out",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as output:
        result=run(args.workspace,Path(output))
    args.out.write_text(json.dumps(result,indent=2)+"\n")
