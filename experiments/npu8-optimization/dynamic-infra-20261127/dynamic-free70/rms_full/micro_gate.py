"""Compare output, dX, and dW bits. Diagnose and skip timing after any mismatch."""
import argparse
import json
from pathlib import Path
import time

import torch
from rms_full import aten_backward, aten_forward, backward_row, candidate, forward_row, stock


def tensor_diff(ref, got):
    ref = ref.detach().cpu().contiguous()
    got = got.detach().cpu().contiguous()
    ref_bits = ref.view(torch.int16).flatten()
    got_bits = got.view(torch.int16).flatten()
    changed = (ref_bits != got_bits).nonzero().flatten()
    diff = {"different": changed.numel(), "numel": ref.numel(),
            "max_abs": float((ref.float() - got.float()).abs().max())}
    if changed.numel():
        i = int(changed[0])
        diff["first"] = {"flat_index": i, "reference_hex": f"{int(ref_bits[i]) & 65535:04x}",
                         "candidate_hex": f"{int(got_bits[i]) & 65535:04x}"}
    return diff


def evaluate(fn, x, w, upstream):
    y = fn(x, w)
    dx, dw = torch.autograd.grad((y.float() * upstream).sum(), (x, w))
    return {"y": y, "dx": dx, "dw": dw}


def compare_stages(ref, candidate_values):
    out = {}
    for name, val in ref.items():
        got = candidate_values[name]
        # A row-scalar is broadcast into the diagnostic vector by the Triton store.
        if val.shape[-1] == 1 and got.shape[-1] == 64:
            got = got[..., :1]
        out[name] = tensor_diff(val, got)
    return {"first_divergence": next((k for k, v in out.items() if v["different"]), None),
            "stages": out}


@torch.no_grad()
def diagnose(x, w, upstream, stock_result):
    ref = aten_forward(x, w, 1e-6)
    dy = upstream.to(dtype=x.dtype)
    ref_bwd = aten_backward(x, w, ref["inv"], ref["z"], dy)
    _, got_inv, got_z, got = forward_row(x, w, 1e-6, debug=True)
    _, _, got_bwd = backward_row(x, w, got_inv, got_z, dy, debug=True)
    _, _, local = backward_row(x, w, ref["inv"], ref["z"], dy, debug=True)
    return {"fwd": compare_stages(ref, got),
            "bwd_with_candidate_forward": compare_stages(ref_bwd, got_bwd),
            "bwd_with_reference_forward": compare_stages(ref_bwd, local),
            "aten_diagnostic_oracle_vs_autograd":
                {"dx": tensor_diff(stock_result["dx"], ref_bwd["dx"]),
                 "dw": tensor_diff(stock_result["dw"], ref_bwd["dw_terms"].sum_to_size(w.shape))}}


def run(device, repeats):
    rng = torch.Generator().manual_seed(1234)
    cases, timing, timed_inputs = [], [], []
    for b, s in ((1, 19), (2, 14), (2, 22), (4, 31)):
        for kind in ("rand_unit_gamma", "rand_mixed_gamma", "near_epsilon", "zero"):
            inp = torch.randn((b, s, 64), generator=rng)
            if kind == "zero":
                inp.zero_()
            if kind == "near_epsilon":
                inp *= .001
            weight = torch.ones(64) if kind == "rand_unit_gamma" else torch.randn(64, generator=rng)
            x = inp.to(dtype=torch.bfloat16, device=device).requires_grad_()
            w = weight.to(dtype=torch.bfloat16, device=device).requires_grad_()
            upstream = torch.randn((b, s, 64), generator=rng).to(device)
            ref, got = evaluate(stock, x, w, upstream), evaluate(candidate, x, w, upstream)
            diffs = {k: tensor_diff(ref[k], got[k]) for k in ref}
            equal = all(not diff["different"] for diff in diffs.values())
            record = {"shape": [b, s], "kind": kind, "equal": equal, "components": diffs}
            if not equal:
                record["diagnosis"] = diagnose(x, w, upstream, ref)
            cases.append(record)
            if kind == "rand_mixed_gamma":
                timed_inputs.append((x, w, upstream))
    passed = all(c["equal"] for c in cases)
    if passed and repeats and device == "npu":
        for i in range(8):
            for fn in (stock, candidate):
                evaluate(fn, *timed_inputs[i % 4])
        for order in ("AB", "BA", "AB", "BA"):
            pair = {}
            for label in order:
                fn = stock if label == "A" else candidate
                torch.npu.synchronize()
                start = time.perf_counter_ns()
                for i in range(repeats):
                    evaluate(fn, *timed_inputs[i % 4])
                torch.npu.synchronize()
                pair[label] = (time.perf_counter_ns() - start) / repeats / 1000
            timing.append({"order": order, "calls": repeats, "stock_us": pair["A"],
                           "candidate_us": pair["B"], "reduction_pct": (1 - pair["B"]/pair["A"]) * 100})
    return {"device": device, "torch": str(torch.__version__), "seed": 1234, "width": 64,
            "bitwise_pass": passed, "cases": cases, "timing": timing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True, choices=("cpu", "npu"))
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=96)
    args = parser.parse_args()
    if args.device == "npu":
        import torch_npu
        torch.npu.set_device(0)
    else:
        torch.set_num_threads(2)
    report = run(args.device, args.repeats)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": report["bitwise_pass"],
                      "equal_cases": sum(c["equal"] for c in report["cases"]), "timing": report["timing"]}))
