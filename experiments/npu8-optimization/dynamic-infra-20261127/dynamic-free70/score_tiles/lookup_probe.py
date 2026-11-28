"""Locate the first BF16 score discrepancy at the cast, table, gather or causal mask."""
import argparse
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
import torch_npu
import triton
import triton.language as tl


@triton.jit
def inspect_words(raw, lut, words, selected, stage):
    length = tl.num_programs(0)
    batch = tl.num_programs(1)
    size = batch * 4 * length * length
    index = (tl.program_id(1) * length + tl.program_id(0)) * 128 + tl.arange(0,128)
    live = index < size
    scale = (tl.load(raw + index, mask=live, other=0).to(tl.float32) * 0.25).to(tl.bfloat16)
    u = scale.to(tl.uint16, bitcast=True).to(tl.int32)
    s = scale.to(tl.int16, bitcast=True).to(tl.int32) & 65535
    f = (scale.to(tl.float32).to(tl.int32, bitcast=True) >> 16) & 65535
    tl.store(words + index, u, mask=live)
    # Each plane is 128-element aligned even for odd sequence lengths.
    stride = (size + 127) // 128 * 128
    tl.store(words + stride + index, s, mask=live)
    tl.store(words + 2*stride + index, f, mask=live)
    y = tl.load(lut + (u & 65535))
    y_signed = tl.load(lut + s)
    y_floatbits = tl.load(lut + f)
    tl.store(selected + index, y, mask=live)
    tl.store(selected + stride + index, y_signed, mask=live)
    tl.store(selected + 2*stride + index, y_floatbits, mask=live)
    row = index // length % length
    col = index % length
    tl.store(stage + index, scale, mask=live)
    tl.store(stage + stride + index, (tl.where(row >= col, y, 0.) / length.to(tl.float32)).to(tl.bfloat16), mask=live)


def diff(a,b):
    left = a.reshape(-1).contiguous()
    right = b.reshape(-1).contiguous()
    if left.dtype == torch.bfloat16:
        left,right = left.view(torch.int16),right.view(torch.int16)
    bad = (left != right).nonzero().flatten()
    return dict(count=bad.numel(), examples=[[int(i), int(left[i]), int(right[i])] for i in bad[:6]])


def run(args):
    sys.path.insert(0, str(args.attention_dir))
    from micro import build
    torch.npu.set_device(0)
    code_sequence = torch.arange(65536, dtype=torch.int32).short().view(torch.bfloat16)
    table = F.silu(code_sequence.npu())
    staged = table.cpu()
    torch.npu.synchronize()
    result = []
    for i,(b,l,layout) in enumerate([(1,13,"matmul"),(2,14,"matmul"),(4,31,"matmul"),(1,13,"cpu_roundtrip")]):
        q,k,_,_ = build(b,l,"npu","finite",250 + (0 if layout=="cpu_roundtrip" else i))
        raw = (q @ k.transpose(-1,-2)).detach()
        if layout == "cpu_roundtrip":
            raw = raw.cpu().contiguous().npu()
        size = raw.numel()
        stride = (size + 127) // 128 * 128
        scaled = raw * 0.25
        silu = F.silu(scaled)
        reference = silu.masked_fill(~torch.ones(l,l,device=raw.device,dtype=torch.bool).tril(),0.) / float(l)
        expected_word = scaled.cpu().contiguous().view(torch.int16).int().reshape(-1) & 65535
        expected_lut = staged.view(torch.int16)[expected_word.long()].contiguous().view(torch.bfloat16)
        words = torch.full((3,stride),-1,dtype=torch.int32,device=raw.device)
        selected = torch.full((3,stride),float("nan"),dtype=torch.bfloat16,device=raw.device)
        stages = torch.full((2,stride),float("nan"),dtype=torch.bfloat16,device=raw.device)
        inspect_words[(l,b,1)](raw,table,words,selected,stages)
        words,selected,stages = (t.cpu()[:,:size] for t in (words,selected,stages))
        direct = silu.cpu().reshape(-1)
        result.append(dict(shape=[b,l], layout=layout, input_stride=list(raw.stride()),
                           storage_bytes=raw.untyped_storage().nbytes(), logical_bytes=raw.numel()*raw.element_size(),
                           formats={name:int(torch.ops.npu.get_npu_format(t)) for name,t in (("raw",raw),("table",table))},
                           integral_keys=[diff(expected_word,lane) for lane in words],
                           scalar_table=diff(direct,expected_lut), gathered=[diff(expected_lut,lane) for lane in selected],
                           scaled=diff(scaled.cpu(),stages[0]), scores=diff(reference.cpu(),stages[1]),
                           pos_lut=diff(expected_lut[expected_word<32768],selected[0][expected_word<32768]),
                           neg_lut=diff(expected_lut[expected_word>=32768],selected[0][expected_word>=32768])))
    args.output.write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    for f in ("attention-dir","output"):
        cli.add_argument("--"+f,required=True,type=Path)
    run(cli.parse_args())
