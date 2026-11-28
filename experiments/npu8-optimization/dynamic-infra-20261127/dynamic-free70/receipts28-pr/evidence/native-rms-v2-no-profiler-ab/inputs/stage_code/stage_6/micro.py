"""Test original native ATen rounding under one C++ autograd node per RMS."""
import argparse
import json
from pathlib import Path
import sys

import torch
from build import build
from block_gate import full_block


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",required=True,choices=("cpu","npu"))
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--build-dir",required=True,type=Path)
    p.add_argument("--reference-dir",type=Path,default=Path(__file__).resolve().parents[1]/"rms_full")
    p.add_argument("--model-file",type=Path,required=True,help="unmodified HSTU workspace model.py")
    p.add_argument("--repeats",type=int,default=96)
    args=p.parse_args()
    if args.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    else:
        torch.set_num_threads(2)
    lib=build(args.build_dir)
    torch.ops.load_library(str(lib))
    op=torch.ops.npu_hstu_native_rms.rms.default
    sys.path.insert(0,str(args.reference_dir.resolve()))
    import micro_gate as gate
    from rms_full import aten_backward, aten_forward
    gate.candidate=lambda x,w:op(x,w,1e-6)
    @torch.no_grad()
    def diagnose(x,w,up,base):
        fw=aten_forward(x,w,1e-6)
        bw=aten_backward(x,w,fw["inv"],fw["z"],up.to(x.dtype))
        row=bw["grad_var"]/64
        dx=bw["direct"] + row*bw["double_x"]
        return {"original_vs_row_divided_before_broadcast":gate.tensor_diff(bw["grad_sq"],row.expand_as(x)),
                "original_dx_vs_native_formula":gate.tensor_diff(base["dx"],bw["dx"]),
                "original_dx_vs_hoisted_ATen":gate.tensor_diff(base["dx"],dx)}
    gate.diagnose=diagnose
    blocks=full_block(args.model_file,op,gate.tensor_diff,args.device)
    report=gate.run(args.device,args.repeats if all(r["equal"] for r in blocks) else 0)
    report["full_HSTU_block"]=blocks
    report["bitwise_pass"]=report["bitwise_pass"] and all(r["equal"] for r in blocks)
    report["implementation"]="native C++ eager ATen / mean gradient row division"
    report["build"]=json.loads((args.build_dir/"rms-build.json").read_text())
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"bitwise":report["bitwise_pass"],"timing":report["timing"]}))


if __name__=="__main__":
    main()
