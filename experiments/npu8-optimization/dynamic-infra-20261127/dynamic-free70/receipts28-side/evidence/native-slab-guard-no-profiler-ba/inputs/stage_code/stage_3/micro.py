"""Replay live changes and time alternating 1,024-table guard evaluations."""
import argparse
from collections import Counter
import importlib.util
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from time import perf_counter_ns
from types import SimpleNamespace

import torch

from adapter import SlabParameterGuard
from stage import stage


def pct(values, fraction):
    return sorted(values)[max(0, math.ceil(len(values) * fraction) - 1)]


def main(args):
    if args.device == "npu":
        import torch_npu
    source = Path(__file__).resolve().parent.parent / "side_slab" / "side_slab.py"
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "side_slab.py"
        shutil.copyfile(source, path)
        stage(Path(tmp))
        spec = importlib.util.spec_from_file_location("staged_slab", path)
        staged = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(staged)
    original = staged.python_parameter_guard
    torch.manual_seed(710)
    tables = torch.nn.ModuleList(torch.nn.Embedding(2, 4, device=args.device) for _ in range(1024))
    prior = tuple(tables)
    weights = tuple(t.weight for t in prior)
    owner = SimpleNamespace(tables=prior, weights=weights, pointers=tuple(w.data_ptr() for w in weights))
    modes = [SlabParameterGuard(Counter(), native=b, directory=args.build) for b in (False, True)]
    results = {}

    def check(name, expected, live=tables):
        outcomes = [g.matches(owner, live, original) for g in modes]
        if outcomes != [expected, expected]:
            raise AssertionError((name, outcomes))
        results[name] = outcomes

    check("valid", True)
    for i in (0, 512, 1023):
        tables[i] = tables[(i + 1) % len(tables)]
        check("replaced_module_" + str(i), False)
        tables[i] = prior[i]
    t, w = tables[-1], weights[-1]
    t.weight = torch.nn.Parameter(w.detach().clone())
    check("replaced_weight", False)
    t.weight = w
    data = w.data
    w.data = w.detach().clone()
    check("rebound_data", False)
    w.data = data
    w.requires_grad_(False)
    check("frozen", False)
    w.requires_grad_(True)
    w.grad = torch.ones_like(w)
    check("materialized_grad", False)
    w.grad = None
    hooks = (
        (t.register_forward_hook, "module_forward"),
        (t.register_forward_pre_hook, "module_forward_pre"),
        (t.register_full_backward_hook, "module_backward"),
        (t.register_full_backward_pre_hook, "module_backward_pre"),
        (w.register_hook, "weight_backward"),
        (w.register_post_accumulate_grad_hook, "weight_post_accumulate"),
    )
    for register, label in hooks:
        handle = register(lambda *a: None)
        check(label, False)
        handle.remove()
    check("shorter", False, torch.nn.ModuleList(prior[:-1]))
    check("public_fallback", True, list(prior))
    check("restored", True)
    warmup, counts = 5, args.repetitions
    elapsed = [[], []]
    for i in range(warmup + counts):
        for leg in ((0, 1) if i % 2 == 0 else (1, 0)):
            tic = perf_counter_ns()
            assert modes[leg].matches(owner, tables, original)
            value = (perf_counter_ns() - tic) / 1000
            if i >= warmup:
                elapsed[leg].append(value)
    aten = {}
    for name, mode in zip(("Python", "native"), modes):
        with torch.autograd.profiler.profile(use_device=None) as prof:
            for _ in range(5):
                assert mode.matches(owner, tables, original)
        aten[name] = {x.key: x.count for x in prof.key_averages() if x.key.startswith("aten::")}
    data = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
            "device": args.device, "cases_A_B": results, "aten_events_in_5_scans": aten,
            "scan_duration_us": {name: {"p50": pct(vals, .5), "p95": pct(vals, .95),
                                        "p99": pct(vals, .99), "values_us": vals}
                                 for name, vals in zip(("Python", "native"), elapsed)},
            "counts": [dict(m.stats) for m in modes]}
    args.output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print([(name, series["p50"], series["p95"]) for name, series in data["scan_duration_us"].items()], aten)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("build", type=Path)
    cli.add_argument("output", type=Path)
    cli.add_argument("--device", choices=("cpu", "npu"), default="cpu")
    cli.add_argument("--repetitions", type=int, default=40)
    main(cli.parse_args())
