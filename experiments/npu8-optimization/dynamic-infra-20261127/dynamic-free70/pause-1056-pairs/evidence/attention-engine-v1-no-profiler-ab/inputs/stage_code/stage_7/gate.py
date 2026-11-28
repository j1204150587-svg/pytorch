"""Whole-block first-order bit equality for the dynamic schedule and timed paired windows."""
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

import torch
from stage import stage


def load_module(model_dir, dest):
    for name in ("model.py", "train_deepspeed.py"):
        shutil.copyfile(model_dir / name, dest / name)
    stage(dest, with_runner=False)
    spec = importlib.util.spec_from_file_location("hstu_attn_gate_model", dest / "model.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    sys.path.insert(0, str(model_dir))
    spec.loader.exec_module(module)
    return module


def sync(device):
    if device == "npu":
        torch.npu.synchronize()


def evaluate(model, x, up):
    y = model(x)
    grads = torch.autograd.grad((y.float() * up).sum(), (x, *model.parameters()))
    return (y.detach(), *(g.detach() for g in grads))


def differences(names, a, b):
    changed = {}
    for name, ref, got in zip(names, a, b, strict=True):
        ref = ref.contiguous().cpu()
        got = got.contiguous().cpu()
        neq = ref.view(torch.int16) != got.view(torch.int16)
        if neq.any():
            changed[name] = {"neq_bf16": int(neq.sum()), "max_abs": float((ref.float()-got.float()).abs().max())}
    return changed


def run(args, slot):
    module = load_module(args.model_dir, slot)
    torch.manual_seed(202611)
    a = module.HSTUBlock(64, 4, 16, 16, 0).bfloat16().to(args.device)
    b = copy.deepcopy(a)
    a._attn_engine, b._attn_engine = False, True
    names = ("out", "dx", *dict(a.named_parameters()))
    print(f"checking {names}", flush=True)
    rows, examples = [], []
    schedule = json.loads(args.schedule.read_text(encoding="utf8"))
    if args.limit:
        schedule = schedule[:args.limit]
    tests = [(i, s, "random_noncontiguous_upstream") for i,s in enumerate(schedule)]
    tests += [(1000+i, s, "zero_input") for i,s in enumerate(schedule[:4])]
    tests += [(2000+i, s, "zero_upstream") for i,s in enumerate(schedule[:4])]
    for i, (size, length), mode in tests:
        torch.manual_seed(17_000+i)
        x = torch.randn(size,length,64).bfloat16().to(args.device)
        up = torch.randn(size,64,length).to(args.device).transpose(1,2)
        if mode == "zero_input":
            x.zero_()
        elif mode == "zero_upstream":
            up.zero_()
        x.requires_grad_()
        y = x.detach().clone().requires_grad_()
        result = differences(names, evaluate(a,x,up), evaluate(b,y,up))
        rows.append({"shape": [size,length], "mode":mode, "different":result})
        if i in (0,3,4,5):
            examples.append((x,y,up))
    enabled_hits, disabled_hits = b._attn_engine_hits, a._attn_engine_hits
    passed = (all(not r["different"] for r in rows) and enabled_hits == len(tests) and disabled_hits == 0)
    # Eval path must use the Python arm when candidate is enabled.
    with torch.no_grad():
        b.eval()
        x = examples[0][0]
        eval_diff = differences(("out",), (a.eval()(x),), (b(x),))
        a.train()
        b.train()
    passed = passed and not eval_diff and b._attn_engine_hits == enabled_hits
    windows = []
    if passed and args.repeats and len(examples)==4:
        for order in ["AB","BA","AB","BA"]:
            pair = {}
            for leg in order:
                candidate = leg == "B"
                model = b if candidate else a
                for x,y,up in examples:
                    evaluate(model, y if candidate else x, up)
                sync(args.device)
                before = time.perf_counter_ns()
                for j in range(args.repeats):
                    x,y,up = examples[j % len(examples)]
                    evaluate(model, y if candidate else x, up)
                sync(args.device)
                pair[leg] = (time.perf_counter_ns() - before) / args.repeats / 1000.0
            windows.append({"order":order, "block_forward_backward_us":pair})
    return {"device":args.device, "torch":str(torch.__version__), "gate_cases":len(rows),
            "different_cases":sum(bool(r["different"]) for r in rows), "compared":list(names),
            "enabled_hits":enabled_hits,"disabled_hits":disabled_hits,"eval_fallback_equal":not eval_diff,
            "bitwise":passed,"cases":rows,"repeats":args.repeats,"timing":windows}


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",choices=("cpu","npu"),required=True)
    p.add_argument("--library",required=True,type=Path)
    p.add_argument("--model-dir",type=Path,required=True)
    p.add_argument("--schedule",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--limit",type=int,default=0)
    p.add_argument("--repeats",type=int,default=32)
    args=p.parse_args()
    if args.device=="npu":
        import torch_npu
        torch.npu.set_device(0)
    else:
        torch.set_num_threads(1)
    os.environ["NPU_ATTENTION_ENGINE_LIBRARY"]=str(args.library.resolve())
    with tempfile.TemporaryDirectory() as td:
        report=run(args,Path(td))
    args.out.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="cases"}),flush=True)
    sys.exit(0 if report["bitwise"] else 2)
