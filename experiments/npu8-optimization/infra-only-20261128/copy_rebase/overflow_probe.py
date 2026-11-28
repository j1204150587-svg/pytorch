"""Standalone single-rank ZeRO-2 probe for finite recovery after non-finite BF16 copies."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("workspace",type=Path)
    parser.add_argument("output",type=Path)
    parser.add_argument("--flag",required=True)
    parser.add_argument("--switch",required=True,choices=("0","1"))
    args=parser.parse_args()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1",0))
        port=sock.getsockname()[1]
    os.environ.update(LOCAL_RANK="0",RANK="0",WORLD_SIZE="1",MASTER_ADDR="127.0.0.1",MASTER_PORT=str(port),
        DS_ACCELERATOR="npu",RECSYS_ZERO2_BATCH_OVERFLOW="1",NPU_DYNAMIC_REBASE_COUNT="1")
    os.environ[args.flag]=args.switch
    import torch
    import torch_npu
    import deepspeed
    sys.path.insert(0,str(args.workspace))
    import overflow
    assert hashlib.sha256(Path(overflow.__file__).read_bytes()).hexdigest()=="b2bb6481d20438ffe07033fab6ea2aa14097cb62c3e7209cb619c503cc427a55"
    torch.manual_seed(1234)
    torch.npu.set_device(0)
    deepspeed.init_distributed(dist_backend="hccl",auto_mpi_discovery=False)
    config=json.loads((args.workspace/"configs/deepspeed_zero2_bf16_overflow_on.json").read_bytes())
    config.update(train_micro_batch_size_per_gpu=1,train_batch_size=1)
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight=torch.nn.Parameter(torch.linspace(-.5,.5,4096))
        def forward(self,x):
            return (self.weight*x).float().mean()
    model=Tiny()
    optim=torch.optim.AdamW(model.parameters(),lr=.01,foreach=False)
    engine,_,_,_=deepspeed.initialize(model=model,optimizer=optim,config=config,dist_init_required=False)
    zero=engine.optimizer
    stats=overflow.install(zero)
    library,={r.split()[-1] for r in Path("/proc/self/maps").read_text().splitlines() if r.split()[-1].endswith("/libtorch_npu.so")}
    counter=ctypes.CDLL(library).npu_inplace_copy_rebase_counter
    counter.argtypes=(ctypes.c_uint,)
    counter.restype=ctypes.c_uint64
    def digest(tensor):
        return hashlib.sha256(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()
    def checkpoint():
        params=zero.single_partition_of_fp32_groups
        return dict(master=[digest(t) for t in params],bf16=[digest(t) for t in zero.bit16_groups_flat],
            adamw=[{k:digest(v) if torch.is_tensor(v) else v for k,v in sorted(zero.optimizer.state.get(t,{}).items())} for t in params])
    probes=[]
    for i,case in enumerate(("finite","inf","finite","nan","finite","-inf","finite")):
        x=torch.full((4096,),float(i+1),device="npu",dtype=torch.bfloat16)
        loss=engine(x)
        engine.backward(loss)
        grad,=[g for group in zero.averaged_gradients.values() for g in group if g is not None]
        assert grad.numel()==4096 and grad.dtype==torch.bfloat16
        host=grad.cpu().clone()
        if case!="finite":host.view(-1)[133]=float(case)
        source=host.npu().reshape(-1)
        scratch_parent=torch.empty(4160,device="npu",dtype=torch.bfloat16)
        scratch=scratch_parent[64:4160]
        before_count=counter(0)
        scratch.copy_(source)
        accepted=counter(0)-before_count
        assert accepted==int(args.switch), (i,args.flag,accepted)
        assert torch.equal(scratch.cpu().view(torch.int16),host.reshape(-1).view(torch.int16))
        grad.reshape(-1).copy_(scratch)
        before=checkpoint()
        pre_skip=int(engine.skipped_steps)
        engine.step()
        torch.npu.synchronize()
        after=checkpoint()
        expected=case!="finite"
        delta=int(engine.skipped_steps)-pre_skip
        assert bool(zero.overflow)==expected and delta==int(expected) and bool(engine.was_step_applied())!=expected
        assert (before==after)==expected and (expected or before["master"]!=after["master"])
        probes.append(dict(case=case,loss=float(loss),target_copy_accepted=accepted,overflow=bool(zero.overflow),
            skip_delta=delta,applied=bool(engine.was_step_applied()),gradient_sha256=digest(host),before=before,after=after))
    assert stats.fast_calls==len(probes) and not stats.fallback_calls and not stats.errors
    args.output.write_text(json.dumps(dict(flag=args.flag,switch=args.switch,library=library,records=probes,
        scan=stats.snapshot()),indent=2)+"\n")
    torch.distributed.destroy_process_group()


if __name__=="__main__":
    main()
