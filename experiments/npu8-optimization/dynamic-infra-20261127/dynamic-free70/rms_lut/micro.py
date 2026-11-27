"""Test the stock-device BF16 lookup across all codes, real shapes, and HSTU residuals."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

import torch
import lut_rms as rms


def import_file(name,path):
    spec=importlib.util.spec_from_file_location(name,str(path))
    mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod)
    return mod


def diff(ref,got):
    ref,got=ref.detach().cpu().contiguous(),got.detach().cpu().contiguous()
    u,v=ref.view(torch.int16).flatten(),got.view(torch.int16).flatten()
    ix=(u!=v).nonzero().flatten()
    result={"numel":u.numel(),"different":ix.numel()}
    if ix.numel():
        i=int(ix[0]); result["first"]=[i,f"{int(u[i])&65535:04x}",f"{int(v[i])&65535:04x}"]
        delta=(ref.float()-got.float()).abs();mask=torch.isfinite(delta)
        result["max_finite_abs"]=float(delta[mask].max()) if mask.any() else None
    return result


def original(x,w,eps=rms.EPS):
    return (x*torch.rsqrt(x.pow(2).mean(-1,True)+eps))*w


def evaluate(fn,x,w,dy):
    y=fn(x,w)
    dx,dw=torch.autograd.grad((y.float()*dy).sum(),(x,w))
    return dict(y=y.detach(),dx=dx,dw=dw)


@torch.no_grad()
def diagnose(x,w,t):
    y,inv,z,values=rms.parts(x,w,t,debug=True)
    sq=x.pow(2); mean=sq.mean(-1,True); native_inv=(mean+rms.EPS).rsqrt(); native_z=x*native_inv
    ref=(sq,mean,native_inv,native_z,native_z*w)
    comparisons={}
    for name,a,b in zip(("square","mean","inv_from_table","z","y"),ref,values):
        if a.shape[-1]==1 and b.shape[-1]!=1:b=b[...,:1]
        comparisons[name]=diff(a,b)
    return comparisons


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",required=True,choices=("npu","cpu"))
    p.add_argument("--build-dir",required=True,type=Path)
    p.add_argument("--model-file",required=True,type=Path)
    p.add_argument("--schedule",required=True,type=Path)
    p.add_argument("--builder-dir",type=Path,default=Path(__file__).resolve().parents[1]/"rms_native")
    p.add_argument("--repeats",type=int,default=96)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    report={"device":args.device,"torch":str(torch.__version__),"timing":[]}
    torch.set_num_threads(2)
    if args.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    sync=torch.npu.synchronize if args.device=="npu" else lambda:None
    synchronize_bytes=(lambda:torch.npu.memory_allocated(0)) if args.device=="npu" else lambda:None
    library=import_file("rms_library_builder",args.builder_dir/"build.py").build(args.build_dir,Path(__file__).with_name("rms_lut_parts.cpp"))
    torch.ops.load_library(str(library))
    shared_a=lambda x,w,eps=rms.EPS:rms.run(x,w,eps,False)
    shared_b=lambda x,w,eps=rms.EPS:rms.run(x,w,eps,True)
    x=torch.zeros((1,1,64),dtype=torch.bfloat16,device=args.device)
    sync();allocated=synchronize_bytes();start=time.perf_counter_ns()
    t=rms.get_table(x,rms.EPS);sync()
    cold=(time.perf_counter_ns()-start)/1000
    allocated_after=synchronize_bytes()
    report["cold_table"]={**rms.BUILDS[0],"synchronized_us":cold,"allocated_before":allocated,"allocated_after":allocated_after}
    all_codes=torch.arange(65536,dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(args.device)
    stock_map=(all_codes.reshape(256,256,1)+rms.EPS).rsqrt().flatten()
    report["all_65536_entries_vs_other_shape"]=diff(stock_map,t)
    # Include raw negative/NaN lookup keys as well as actual mean values.
    tiny=[]
    for b,l in ((1,19),(2,14),(2,22),(4,31)):
        keys=torch.linspace(0,65535,b*l,dtype=torch.float64).to(torch.int64)
        fixed=[0x0000,0x8000,0x0001,0x8001,0x7f80,0xff80,0x7fc1,0xffc1,0x7f81,0x3f80,0xbf80]
        keys[:len(fixed)]=torch.tensor(fixed,dtype=torch.int64)
        m=keys.to(torch.int16).view(torch.bfloat16).reshape(b,l,1).to(args.device)
        expected=(m+rms.EPS).rsqrt()
        tiny.append({"shape":[b,l],"diff":diff(expected,t[keys.to(args.device)].reshape(b,l,1))})
    report["scalar_map_on_actual_shapes"]=tiny
    gen=torch.Generator().manual_seed(1234)
    inputs=[];cases=[]
    schedule=json.loads(args.schedule.read_bytes())
    named={(1,19),(2,14),(2,22),(4,31)}
    for b,l in schedule:
        kinds=("unit","mixed","near_epsilon","zero","limits","nonfinite") if (b,l) in named else ("mixed",)
        for kind in kinds:
            cpu=torch.randn((b,l,64),generator=gen).to(torch.bfloat16)
            if kind=="near_epsilon":cpu*=.001
            if kind=="zero":cpu.zero_()
            if kind in ("limits","nonfinite"):
                patterns = ([0,0x8000,1,0x8001,0x007f,0x807f,0x0080,0x7f7f,0xff7f]
                            if kind=="limits" else [0,0x8000,0x7f80,0xff80,0x7fc1,0xffc1,0x7f81,0xff81])
                for row,pattern in enumerate(patterns):
                    cpu.view(torch.int16).reshape(-1,64)[row,1]=pattern if pattern<32768 else pattern-65536
            x=cpu.to(args.device).requires_grad_()
            wc=torch.ones(64) if kind in ("unit","limits","nonfinite") else torch.randn(64,generator=gen)
            w=wc.to(device=args.device,dtype=torch.bfloat16).requires_grad_()
            dy=torch.randn(x.shape,generator=gen).to(args.device)
            ref,base,alt=[evaluate(fn,x,w,dy) for fn in (original,shared_a,shared_b)]
            ac={k:diff(ref[k],base[k]) for k in ref}
            bc={k:diff(ref[k],alt[k]) for k in ref}
            ab={k:diff(base[k],alt[k]) for k in ref}
            equal=all(not v["different"] for z in (ac,bc) for v in z.values())
            row={"shape":[b,l],"kind":kind,"equal":equal,"A_vs_original":ac,"B_vs_original":bc,"B_vs_A":ab}
            if not equal:
                row["forward_diagnosis"]=diagnose(x,w,t)
            cases.append(row)
            if (b,l) in named and kind=="mixed":inputs.append((x,w,dy))
    report["cases"]=cases
    blocks=import_file("native_rms_full_block",args.builder_dir/"block_gate.py")
    report["HSTU_blocks"]={label:blocks.full_block(args.model_file,fn,diff,args.device) for label,fn in (("A",shared_a),("B",shared_b))}
    # Invalid HSTU epsilon always uses the native path. An alternate stream
    # must wait for the already-built immutable table, with equivalent results.
    x,w,dy=inputs[0];fallbacks=[]
    for eps in (0.0,-1e-6,1e-4):
        before=rms.COUNTS["fallback"]
        expected=shared_a(x,w,eps);actual=shared_b(x,w,eps)
        fallbacks.append({"eps":eps,"fallback_count":rms.COUNTS["fallback"]-before,"result":diff(expected,actual)})
    report["epsilon_fallback"]=fallbacks
    if args.device=="npu":
        sync(); stream=torch.npu.Stream();before=rms.COUNTS["other_stream_wait"]
        with torch.npu.stream(stream):
            ref=shared_a(x,w);other=shared_b(x,w);stream.synchronize()
        report["other_stream"]={"waits":rms.COUNTS["other_stream_wait"]-before,"result":diff(ref,other)}
    passed=(not report["all_65536_entries_vs_other_shape"]["different"] and all(not c["diff"]["different"] for c in tiny)
            and all(c["equal"] for c in cases) and all(c["equal"] for leg in report["HSTU_blocks"].values() for c in leg)
            and all(c["fallback_count"]==1 and not c["result"]["different"] for c in fallbacks))
    if args.device=="npu":
        passed=passed and report["other_stream"]["waits"]==1 and not report["other_stream"]["result"]["different"]
    report["bitwise_pass"]=passed
    if passed and args.device=="npu" and args.repeats:
        for i in range(8):
            for fn in (shared_a,shared_b):evaluate(fn,*inputs[i%len(inputs)])
        for order in ("AB","BA","AB","BA"):
            pair={}
            for label in order:
                fn=shared_a if label=="A" else shared_b
                sync();start=time.perf_counter_ns()
                for i in range(args.repeats):evaluate(fn,*inputs[i%len(inputs)])
                sync();pair[label]=(time.perf_counter_ns()-start)/1000/args.repeats
            report["timing"].append({"order":order,"native_us":pair["A"],"lut_us":pair["B"],"reduction_pct":(1-pair["B"]/pair["A"])*100})
    report["counts"]=dict(rms.COUNTS)
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"passed":passed,"cases":len(cases),"timing":report["timing"]}))


if __name__=="__main__":
    main()
