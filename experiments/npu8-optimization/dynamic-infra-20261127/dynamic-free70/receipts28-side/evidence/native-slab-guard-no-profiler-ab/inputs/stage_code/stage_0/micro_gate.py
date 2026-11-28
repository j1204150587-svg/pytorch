"""Check dynamic odd/even widths, repeated IDs, independent table gradients and BF16 rounding."""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
import side_tables_op


def bits(t):
    return t.detach().contiguous().view(torch.int16)


def check_shape(tables, batch, length, device, generator):
    rows, width, features = 64, 64, 4
    cpu_weights = torch.randn((tables, rows, width), generator=generator).to(torch.bfloat16)
    weights = [torch.nn.Parameter(x.clone().to(device)) for x in cpu_weights.unbind(0)]
    ids = torch.randint(rows, (batch, length, tables, features), generator=generator)
    # Every table has duplicate IDs within a token and across tokens. Include
    # patterns at the row and table boundaries that are especially easy to mix.
    ids[..., 1] = ids[..., 0]
    ids[..., 2] = 0
    ids[..., 3] = rows - 1
    ids = ids.to(device)
    upstream = torch.randn((batch, length, width), generator=generator).to(device=device, dtype=torch.bfloat16)
    offsets = (torch.arange(tables, dtype=torch.int64) * rows).view(-1,1,1,1).to(device)
    ref = None
    for i, w in enumerate(weights):
        value = F.embedding(ids[:, :, i, :], w).mean(-2)
        ref = value if ref is None else ref + value
    candidate = torch.ops.npu_hstu_experiment.grouped_side_sum(weights, ids, offsets)
    a = torch.autograd.grad((ref.float() * upstream.float()).sum(), weights)
    b = torch.autograd.grad((candidate.float() * upstream.float()).sum(), weights)
    expected, observed = torch.cat(a).cpu(), torch.cat(b).cpu()
    grad_match = (bits(expected) == bits(observed)).view(tables, -1).all(1)
    return dict(tables=tables, shape=[batch,length], forward_equal=torch.equal(bits(ref).cpu(), bits(candidate).cpu()),
                returned_grads=len(b), gradients_equal=bool(grad_match.all()),
                changed_table_count=int((~grad_match).sum()), first_changed=(~grad_match).nonzero().flatten()[:5].tolist(),
                max_gradient_error=float((expected.float()-observed.float()).abs().max()))


def rounding_case(device):
    means = torch.tensor([256, 1, -256], dtype=torch.bfloat16, device=device)
    means = means[:,None,None,None].expand(3,1,13,64).contiguous().requires_grad_(True)
    result = side_tables_op.OrderedTableSum.apply(means)
    grads, = torch.autograd.grad(result.float().sum(), means)
    return bool((result == 0).all()) and bool((grads == 1).all())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("npu", "cpu"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--tables", type=int, default=1024)
    args = parser.parse_args()
    if args.device == "npu":
        import torch_npu
        torch.npu.set_device(0)
    else:
        torch.set_num_threads(2)
    generator = torch.Generator(device="cpu").manual_seed(1234)
    results = [check_shape(args.tables,b,l,args.device,generator) for b,l in ((1,19),(2,14),(2,22),(4,31))]
    rounding = rounding_case(args.device)
    passed = rounding and all(x["forward_equal"] and x["gradients_equal"] and x["returned_grads"] == args.tables
                              for x in results)
    output = dict(torch=str(torch.__version__), device=args.device, ordered_rounding=rounding, cases=results, passed=passed)
    args.out.write_text(json.dumps(output, indent=2) + "\n", encoding="utf8")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
