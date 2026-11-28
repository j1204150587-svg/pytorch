"""Profile both Addmm descriptor layouts against the very same inputs and physical stream."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from probe_128 import pair


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--layout",type=Path,required=True)
    cli.add_argument("--output",type=Path,required=True)
    args=cli.parse_args()
    torch.npu.set_device(0)
    torch.set_num_threads(2)
    geo=json.loads(args.layout.read_bytes())
    gen=torch.Generator().manual_seed(1234)
    flat=torch.randn(geo["flat"]["elements"],generator=gen).bfloat16().npu()
    bank=[(pair(flat,r["bias"]),pair(flat,r["transposed_weight"])) for r in geo["projection_views"]]
    inputs={n:{k:torch.randn(n,k,generator=gen).bfloat16().npu() for k in (64,192)} for n in (19,28)}
    torch.npu.synchronize()
    p=torch_npu.profiler
    cfg=p._ExperimentalConfig(profiler_level=p.ProfilerLevel.Level1)
    report={}
    output={}
    with p.profile(activities=[p.ProfilerActivity.CPU,p.ProfilerActivity.NPU],record_shapes=True,experimental_config=cfg,
                   on_trace_ready=p.tensorboard_trace_handler(str(args.output.parent/"profile"))) as prof:
        for step,n,order in ((4,19,"AB"),(5,28,"BA")):
            for name in order:
                k=int(name=="B")
                with torch.profiler.record_function(f"Projection_{name}_{step}"):
                    output[name,step]=[torch.addmm(b[k],inputs[n][w[k].size(0)],w[k]) for b,w in bank]
                    torch.npu.synchronize()
    for step in (4,5):
        report[str(step)]=all(torch.equal(x.cpu().view(torch.int16),y.cpu().view(torch.int16))
                              for x,y in zip(output["A",step],output["B",step]))
    args.output.write_text(json.dumps(report,indent=2))
