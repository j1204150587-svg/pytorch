"""Test an entire HSTU block with the original ATen autograd and paired dynamic timings."""
import argparse
from collections import Counter
import copy
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def native(m,x):
    return torch.ops.npu_hstu_native_block.block.default(x,m.input_norm.weight,m.uvqk.weight,m.uvqk.bias,
            m.output_norm.weight,m.output.weight,m.output.bias,m.input_norm.eps,m.output_norm.eps)


def run(fn,block,x,dy):
    y=fn(block,x)
    grads=torch.autograd.grad((y.float()*dy).sum(),(x,*block.parameters()))
    return y,*grads


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cpu","npu"),required=True)
    parser.add_argument("--model-file",required=True,type=Path)
    parser.add_argument("--build-dir",required=True,type=Path)
    parser.add_argument("--builder",type=Path,default=Path(__file__).resolve().parents[1]/"rms_native/build.py")
    parser.add_argument("--repeats",type=int,default=32)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    if args.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    torch.set_num_threads(2)
    lib=import_file("hstu_shared_builder",args.builder).build(args.build_dir,Path(__file__).with_name("host_block.cpp"))
    torch.ops.load_library(str(lib))
    module=import_file("original_hstu_block",args.model_file)
    stock=lambda m,x:m(x)
    cases=[]; timed=[]; timing=[]
    for b,l in ((1,19),(2,14),(2,22),(4,31)):
        torch.manual_seed(2000+b+l)
        a=module.HSTUBlock(64,4,16,16,0.0).to(dtype=torch.bfloat16,device=args.device)
        c=copy.deepcopy(a)
        x=torch.randn((b,l,64)).to(dtype=torch.bfloat16,device=args.device).requires_grad_()
        z=x.detach().clone().requires_grad_()
        dy=torch.randn((b,l,64)).to(args.device)
        base=run(stock,a,x,dy)
        got=run(native,c,z,dy)
        checks={}
        for n,v,w in zip(("y","dx",*dict(a.named_parameters())),base,got):
            v,w=v.detach().cpu().contiguous(),w.detach().cpu().contiguous()
            checks[n]={"different":int((v.view(torch.int16)!=w.view(torch.int16)).sum()),
                       "elements":v.numel(),"max_abs":float((v.float()-w.float()).abs().max())}
        cases.append({"shape":[b,l],"equal":all(not v["different"] for v in checks.values()),"checks":checks})
        timed.append(((stock,a,x,dy),(native,c,z,dy)))
    passed=all(c["equal"] for c in cases)
    report={"device":args.device,"bitwise_pass":passed,"cases":cases,"timing":timing,
            "build":json.loads((args.build_dir/"host_block-build.json").read_text())}
    if args.device=="cpu":
        traces=[]
        for inp in timed[1]:
            with torch.autograd.profiler.profile(use_cpu=True) as p:
                run(*inp)
            traces.append([e.name for e in p.function_events if e.name.startswith("aten::")])
        a,b=traces
        report["cpu_aten_sequence"]={"counts":[len(a),len(b)],"identical":a==b,
                "first_mismatch":next(((i,x,y) for i,(x,y) in enumerate(zip(a,b)) if x!=y),None),
                "A_only":dict(Counter(a)-Counter(b)),"B_only":dict(Counter(b)-Counter(a))}
    if passed and args.device=="npu" and args.repeats:
        for pair in timed*2:
            for inp in pair:run(*inp)
        for order in ("AB","BA","AB","BA"):
            pair={}
            for label in order:
                index=0 if label=="A" else 1
                torch.npu.synchronize()
                start=time.perf_counter_ns()
                for i in range(args.repeats):
                    run(*timed[i%len(timed)][index])
                torch.npu.synchronize()
                pair[label]=(time.perf_counter_ns()-start)/1000/args.repeats
            timing.append({"order":order,"stock_us":pair["A"],"candidate_us":pair["B"],
                           "reduction_pct":(1-pair["B"]/pair["A"])*100})
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"pass":passed,"timing":timing}))


if __name__=="__main__":
    main()
