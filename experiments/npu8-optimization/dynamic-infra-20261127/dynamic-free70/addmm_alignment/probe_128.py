"""Exercise compact, 128-byte-aligned Addmm descriptors on actual ZeRO parameter layouts."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch
import torch_npu


def pair(flat, geom):
    shape = geom["shape"]
    stride = geom["stride"]
    start = geom["storage_element_offset"]
    original = flat.as_strided(shape, stride, start)
    addr = original.data_ptr()
    assert addr % 128 == 0
    size = original.numel() * original.element_size()
    storage = flat.untyped_storage()[start*flat.element_size():start*flat.element_size()+size]
    new = torch.empty(0, device=flat.device, dtype=flat.dtype).set_(storage, 0, shape, stride)
    assert new.data_ptr() == addr and storage.nbytes() == size and new.storage_offset() == 0
    return original, new


def main(args, report):
    torch.npu.set_device(0)
    g = torch.Generator().manual_seed(1234)
    geometry = json.loads(args.layout.read_bytes())
    schedule = json.loads(args.schedule.read_bytes())
    layout = geometry["projection_views"]
    total = geometry["flat"]["elements"]
    # Separate storage owners detect stale bindings after alternating real weights.
    owners = [torch.randn(total, generator=g).bfloat16().npu() for _ in range(2)]
    banks = []
    for flat in owners:
        rows = []
        for layer in layout:
            w = pair(flat, layer["transposed_weight"])
            bias = pair(flat, layer["bias"])
            rows.append((w,bias))
        banks.append(rows)
    report["residues_mod_512"] = [[(w[0].data_ptr()%512,b[0].data_ptr()%512) for w,b in bank] for bank in banks]
    streams = (torch.npu.current_stream(),torch.npu.Stream())
    torch.npu.synchronize()
    def inputs(b,l):
        return {k:torch.randn(b*l,k,generator=g).bfloat16().npu() for k in (64,192)}
    def execute(bank,x,leg):
        return [torch.addmm(bias[leg],x[w[leg].size(0)],w[leg]) for w,bias in bank]
    report.update(cases=[],windows=[])
    if args.cold:
        k = int(args.cold=="B")
        slices = [inputs(*s) for s in schedule]
        if args.profile:
            out = args.output.parent/(args.output.stem+"-profile")
            profiler = torch_npu.profiler
            cfg = profiler._ExperimentalConfig(profiler_level=profiler.ProfilerLevel.Level1)
            with profiler.profile(activities=[profiler.ProfilerActivity.CPU,profiler.ProfilerActivity.NPU],
                                  record_shapes=True,experimental_config=cfg,
                                  on_trace_ready=profiler.tensorboard_trace_handler(str(out))) as prof:
                for i in (4,5):
                    with torch.profiler.record_function(f"AddmmProbe_{i}"):
                        execute(banks[i%2],slices[i],k)
                        torch.npu.synchronize()
                prof.step()
            report["profile"]=str(out)
        else:
            for i in range(10):
                execute(banks[i%2],slices[i],k)
            torch.npu.synchronize()
            start=time.perf_counter_ns()
            for i in range(10,60):
                execute(banks[i%2],slices[i],k)
            torch.npu.synchronize()
            report["us_per_eight_addmm"]=(time.perf_counter_ns()-start)/50000
        return
    for step,(b,l) in enumerate(schedule):
        with torch.npu.stream(streams[step%2]):
            x = inputs(b,l)
            baseline = execute(banks[step%2],x,0)
            compact = execute(banks[step%2],x,1)
            diffs = [int((a.cpu().view(torch.int16)!=c.cpu().view(torch.int16)).sum()) for a,c in zip(baseline,compact)]
            report["cases"].append(dict(shape=[b,l],differing_by_layer=diffs))
            if any(diffs):
                return
    report["exact"] = len(report["cases"])==60
    slices = [inputs(*s) for s in schedule]
    shape_window = list(range(10,60))
    for order in ("AB","BA","AB","BA"):
        row = {"order":order}
        for leg in order:
            k = int(leg=="B")
            for i in range(10):
                execute(banks[i%2],slices[i],k)
            torch.npu.synchronize()
            start = time.perf_counter_ns()
            for i in shape_window:
                execute(banks[i%2],slices[i],k)
            torch.npu.synchronize()
            row[leg]=(time.perf_counter_ns()-start)/len(shape_window)/1000
        report["windows"].append(row)
    report["windows_unit"]="us per eight addmm, warmed geometry"
    report["data_ptrs_identical"]=True
    report["cold_children"]=[]
    for i,leg in enumerate("ABBAAB"):
        child = args.output.parent/f"cold-{i}-{leg}.json"
        cmd=[sys.executable,"-B",__file__,"--layout",str(args.layout),"--schedule",str(args.schedule),
             "--output",str(child),"--cold",leg]
        if i>=4:
            cmd.append("--profile")
        subprocess.run(cmd,check=True)
        report["cold_children"].append(child.name)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    for field in ("layout","schedule","output"):
        cli.add_argument("--"+field,required=True,type=Path)
    cli.add_argument("--cold",choices=("A","B"))
    cli.add_argument("--profile",action="store_true")
    args=cli.parse_args()
    data={"exact":False}
    try:
        main(args,data)
    finally:
        args.output.write_text(json.dumps(data,indent=2))
