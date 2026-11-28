"""Check dynamic dense ID prep on the two real streams, then time the complete routes."""
import argparse
import json
from pathlib import Path
import statistics
import time

import torch
import torch_npu

from operator_impl import counts, dense, prepare, witnesses
from original_probe import diff, inputs, SHAPES
from side_ids_impl import reference


def run(args, report):
    torch.npu.set_device(0)
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(1234)
    schedule = json.loads(args.schedule.read_bytes())
    assert len(schedule) == len(set(map(tuple, schedule))) == 60
    s0, s1 = torch.npu.current_stream(), torch.npu.Stream()
    report.update(shapes=schedule, cases=[], windows=[], profile=[])
    for dtype in (torch.int32, torch.int64):
        for i, (b, l) in enumerate(schedule):
            ids, off = inputs(b, l, dtype, generator)
            expected = reference(ids, off, 64)
            x, o = ids.npu(), off.npu()
            stream = (s0, s1)[i % 2]
            stream.wait_stream(s0)
            row = {"shape": [b,l], "dtype": str(dtype), "stream_index": i%2, "routes": {}}
            report["cases"].append(row)
            with torch.npu.stream(stream):
                for mode in "ASP":
                    before = counts.copy()
                    y = prepare(x, o, 64, mode)
                    stream.synchronize()
                    different = diff(y, expected)
                    row["routes"][mode] = {"diff": different, "counts": dict(counts-before)}
                    if different:
                        raise ValueError(f"input mismatch at {b}x{l} {dtype} {mode}")
    report["correctness_counts"] = dict(counts)
    report["plans"] = witnesses
    if counts["returned"] != 118:
        raise ValueError(f"planned path did not register: {witnesses}")
    host, offsets = inputs(2, 22, torch.int64, generator)
    x, o = host.npu(), offsets.npu()
    variations = [("strided_input",x.transpose(0,1),o,64), ("unsupported_rows",x,o,63),
                  ("unsupported_tables",x[:,:,:512].contiguous(),o[:512],64),
                  ("strided_offsets",x,torch.stack([o,o],dim=1)[:,0],64)]
    report["fallbacks"] = []
    for name, v, t, rows in variations:
        before = counts.copy()
        actual = prepare(v,t,rows,"P")
        delta = diff(actual,reference(v,t,rows))
        changed = dict(counts-before)
        report["fallbacks"].append({"case":name,"diff":delta,"counts":changed})
        if delta or not any(k.startswith("fallback_") for k in changed):
            raise ValueError(name)
    del variations,v,t,actual
    expected = reference(host, offsets,64)
    s1.wait_stream(s0)
    with torch.npu.stream(s1):
        y = prepare(x,o,64,"P")
    del x,o
    junk = [torch.full(host.shape,i,dtype=host.dtype,device="npu") for i in range(8)]
    s1.synchronize()
    report["early_release_diff"] = diff(y,expected)
    if report["early_release_diff"]:
        raise ValueError("cross-stream lifetime")
    del junk
    sets = []
    for b,l in SHAPES:
        x,o = inputs(b,l,torch.int64,generator,False)
        x,o = x.npu(),o.npu()
        t = dense(x,o,"P")
        sets.append((x,o,t))
    def submit(route, item):
        x,o,t = item
        if route == "kernel":
            return dense(x,o,"P")
        if route == "transpose":
            return t.movedim(2,0).contiguous()
        return prepare(x,o,64,route)
    for item in sets:
        for mode in ("A","S","P","kernel","transpose"):
            submit(mode,item)
    for order in ("ASP","PSA","ASP","PSA",("kernel","transpose"),("transpose","kernel")):
        legs = {}
        for mode in order:
            before=counts.copy()
            host=[]
            torch.npu.synchronize()
            start=time.perf_counter_ns()
            for i in range(48):
                begin=time.perf_counter_ns()
                y=submit(mode,sets[i%4])
                host.append((time.perf_counter_ns()-begin)/1000)
            torch.npu.synchronize()
            legs[mode]={"calls":48,"per_call_us":(time.perf_counter_ns()-start)/48000,
                        "host_us":host,"host_p50_us":statistics.median(host),"counts":dict(counts-before)}
        report["windows"].append({"order":list(order),"legs":legs})
    if args.profile:
        p = torch_npu.profiler
        with p.profile(activities=[p.ProfilerActivity.CPU,p.ProfilerActivity.NPU],
                       experimental_config=p._ExperimentalConfig(profiler_level=p.ProfilerLevel.Level1),
                       on_trace_ready=p.tensorboard_trace_handler(str(args.profile))):
            for i,(b,l) in enumerate(SHAPES):
                for rep,order in enumerate(("ASP","PSA","ASP")):
                    for mode in order:
                        tag=f"DenseSideIDs/{b}x{l}/{rep}/{mode}"
                        before=counts.copy()
                        torch.npu.synchronize()
                        with torch.autograd.profiler.record_function(tag):
                            y=submit(mode,sets[i])
                            torch.npu.synchronize()
                        report["profile"].append({"tag":tag,"counts":dict(counts-before)})
    report["counts"] = dict(counts)
    report["status"] = "passed"


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("schedule","out"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--profile",type=Path)
    args=parser.parse_args()
    result={"status":"incomplete"}
    try:
        run(args,result)
    finally:
        args.out.write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
