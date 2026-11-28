"""Exercise original eager NPU view through its public entry on real shapes."""
import argparse
import json
import os
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("work", type=Path)
    parser.add_argument("switch", choices=("0", "1"))
    args = parser.parse_args()
    os.environ.update(TORCH_NPU_VIEW_INLINE_DIMS=args.switch, TORCH_NPU_VIEW_INLINE_DIMS_AUDIT="1")
    import torch
    import torch_npu
    torch.npu.set_device(0)
    sys.path.insert(0, str(args.work))
    import overflow
    schedule = json.loads((args.work / "configs/dynamic_shapes_npu8.json").read_bytes())
    assert len(set(map(tuple,schedule))) == 60
    observed = []
    for b,l in schedule:
        n = b*l
        ref = torch.arange(n+4, dtype=torch.float32)
        flat = ref.npu()
        base = flat[2:2+n]
        view = base.view(b,-1,1)
        expected = ref[2:2+n].view(b,-1,1)
        before = flat._version
        assert tuple(view.stride()) == tuple(expected.stride()) and view.storage_offset() == 2
        assert torch._C._is_alias_of(flat,view) and view._is_view() and torch.equal(view.cpu(), expected)
        view.add_(1)
        expected.add_(1)
        assert (base._version,flat._version,view._version) == (before+1,)*3 and torch.equal(flat.cpu(),ref)
        x = torch.arange(n,dtype=torch.float32).npu().requires_grad_()
        x.view(b,l).sum().backward()
        assert torch.equal(x.grad.cpu(),torch.ones(n))
        observed.append([b,l,list(view.stride()),view.storage_offset(),view._version-before])
    special = []
    owner = torch.arange(36,dtype=torch.float32).npu()
    for name,src,size in [("scalar",owner[0],()),("empty",owner[:0],(2,0)),
                          ("noncontiguous",owner.view(6,6).t(),(6,6)),
                          ("singleton_noncanonical",owner.as_strided((1,4),(9,2),3),(2,2)),
                          ("rank_above_inline",owner[:2],(1,1,1,1,1,1,1,1,1,2))]:
        y=src.view(size)
        assert torch._C._is_alias_of(src,y)
        special.append([name,list(y.shape),list(y.stride()),y.storage_offset(),y.cpu().flatten().tolist()])
    errors = []
    for name, src, size in [("numel",owner,(5,)),("two_infer",owner,(-1,-1)),
                            ("negative",owner,(-2,18)),("empty_ambiguous",owner[:0],(0,-1)),
                            ("nonviewable",owner.view(6,6).t(),(-1,))]:
        try:
            src.view(size)
        except RuntimeError as err:
            errors.append([name,str(err).splitlines()[0]])
        else:
            raise AssertionError(name)
    # Exercise finite/Inf/NaN decisions of the existing 01 scanner on gradient
    # views. The finite update and deliberate skip here are independent probes.
    steps = []
    parameter = torch.tensor([3.],device="npu")
    for kind, value in (("finite",1.),("inf",float("inf")),("nan",float("nan"))):
        values = torch.full((8,),value,dtype=torch.bfloat16,device="npu")
        shards = [values[:4].view(2,2),values[4:].view(1,4)]
        stats = overflow.ScanStats()
        bad = overflow.batched_scan([shards],values.device,lambda t:(~torch.isfinite(t)).sum(),
                                    lambda: (_ for _ in ()).throw(AssertionError("unexpected fallback")),stats,
                                    storage_format=lambda x:int(torch_npu.get_npu_format(x)))
        before = parameter.clone()
        if not bad.item():
            parameter.add_(1)
        skip = torch.equal(before.cpu(),parameter.cpu())
        assert skip == (kind != "finite") and stats.fast_calls == 1 and not stats.fallback_calls
        steps.append([kind,bool(bad.item()),skip,parameter.item()])
    torch.npu.synchronize()
    args.output.write_text(json.dumps(dict(switch=args.switch,shapes=observed,special=special,errors=errors,
                                         overflow=steps),indent=2)+"\n")


if __name__ == "__main__":
    main()
