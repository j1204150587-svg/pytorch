"""Check RMSNorm output/x-grad/weight-grad bits and time paired eager calls."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
from rms_candidate import partial, stock, stock_npu


def result(fn, x, w, upstream):
    out = fn(x,w)
    # The scalarization gives an arbitrary BF16 upstream while keeping a common
    # forward/backward path in both legs (and allowing the offline CPU oracle).
    dx,dw = torch.autograd.grad((out.float()*upstream).sum(),(x,w))
    return (out.detach(),dx,dw)


def compare(a,b):
    out = {}
    for name,x,y in zip(("out","dx","dw"),a,b):
        x,y=x.detach().cpu().contiguous(),y.detach().cpu().contiguous()
        same = x.view(torch.int16) == y.view(torch.int16)
        out[name] = {"numel":x.numel(),"different":int((~same).sum()),
                     "max_abs":float((x.float()-y.float()).abs().max())}
    return out


def run(device, repeats):
    sync = torch.npu.synchronize if device=="npu" else lambda:None
    shapes = ((1,19),(2,14),(2,22),(4,31))
    rng=torch.Generator().manual_seed(1234)
    candidates = {"two_multiply_fusion":partial}
    if device=="npu":
        candidates["npu_rms_norm"]=stock_npu
    cases, data = [],[]
    for batch,length in shapes:
        for kind in ("rand_unit_gamma","rand_mixed_gamma","near_epsilon","zero"):
            x_cpu=torch.randn((batch,length,64),generator=rng)
            if kind=="zero":x_cpu.zero_()
            if kind=="near_epsilon":x_cpu*=.001
            w_cpu=torch.ones(64) if kind=="rand_unit_gamma" else torch.randn(64,generator=rng)
            x=x_cpu.to(device=device,dtype=torch.bfloat16).requires_grad_(True)
            w=w_cpu.to(device=device,dtype=torch.bfloat16).requires_grad_(True)
            upstream=torch.randn((batch,length,64),generator=rng).to(device)
            base=result(stock,x,w,upstream)
            for name,fn in candidates.items():
                checks=compare(base,result(fn,x,w,upstream))
                cases.append(dict(shape=[batch,length],kind=kind,method=name,components=checks,
                                  equal=all(z["different"]==0 for z in checks.values())))
            if kind=="rand_mixed_gamma":
                data.append((x,w,upstream))
    passing = {k:all(c["equal"] for c in cases if c["method"]==k) for k in candidates}
    times = []
    for name,fn in candidates.items():
        if not passing[name] or repeats==0:
            continue
        for i in range(8):
            result(stock,*data[i%len(data)])
            result(fn,*data[i%len(data)])
        for order in ("AB","BA","AB","BA"):
            pair={}
            for label in order:
                measured = stock if label=="A" else fn
                sync()
                start=time.perf_counter_ns()
                for i in range(repeats):
                    result(measured,*data[i%len(data)])
                sync()
                pair[label]=(time.perf_counter_ns()-start)/repeats/1000
            times.append(dict(method=name,order=order,calls=repeats,stock_us_per_call=pair["A"],
                              candidate_us_per_call=pair["B"],reduction_pct=(1-pair["B"]/pair["A"])*100))
    return dict(device=device,torch=str(torch.__version__),shapes=shapes,profiler=False,cases=cases,
                bitwise_pass=passing,timing=times)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("npu","cpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--repeats",type=int,default=96)
    args=parser.parse_args()
    if args.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    else:torch.set_num_threads(2)
    report=run(args.device,args.repeats)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"bitwise_pass":report["bitwise_pass"],"timing":report["timing"]}))
