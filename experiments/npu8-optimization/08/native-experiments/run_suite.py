"""Run the 08 PTA controls and candidates serially on logical NPU 8."""

import argparse
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import subprocess

from run_npu8 import DOCKER, LOCK, capture, check_card, inspect_container
from summarize_pairs import profile, steady_state, training


HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE.parent
INNER = f"/artifacts/{HERE.name}"
VARIANTS = {
    "reference": ("0", "0"),
    "queued": ("1", "0"),
    "flat": ("0", "1"),
    "combined": ("1", "1"),
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def target_usage(out, when):
    card = read(out / f"{when}.card.json")
    data = capture(["npu-smi", "info", "-t", "usages", "-i", str(card["board"]), "-c", str(card["chip"])],
                   timeout=20)
    (out / f"{when}.usages.log").write_text(data.stdout + data.stderr)
    assert data.returncode == 0
    rates = {name: int(value) for name, value in re.findall(r"([A-Za-z]+) Usage Rate\(%\)\s*:\s*(\d+)", data.stdout)}
    assert all(rates[name] == 0 for name in ("Aicore", "Aivector", "Aicpu", "Aicube")), rates
    return rates["HBM"]


def command(container, variant, package, argv, stock=False):
    reference = read(HERE / "steady-command.json")["env"]
    queued, flat = VARIANTS[variant]
    env = dict(reference, RECSYS_ZERO2_BATCH_NORM="1", NPU_EXPERIMENT_QUEUE_COPY_REPLAY=queued,
               NPU_EXPERIMENT_FLAT_ND=flat)
    if not stock:
        env["PYTHONPATH"] = f"{INNER}/{package}"
    flags = [item for name, value in env.items() for item in ("--env", f"{name}={value}")]
    return [*DOCKER, "exec", "--workdir", f"{INNER}/work", *flags, container, *argv]


def invoke(control, out, variant, argv, stock=False, verify_idle=True):
    container, init_pid, package = control
    out.mkdir()
    before = after = None
    if verify_idle:
        check_card(out, "before", container, init_pid)
        before = target_usage(out, "before")
    command_line = command(container, variant, package, argv, stock)
    write(out / "command.json", {"variant": variant, "argv": command_line})
    with (out / "run.log").open("wb") as log:
        completed = subprocess.run(command_line, stdout=log, stderr=subprocess.STDOUT, timeout=320)
    (out / "train.exit").write_text(f"{completed.returncode}\n")
    if verify_idle:
        check_card(out, "after", container, init_pid)
        after = target_usage(out, "after")
        print(out.parent.name, out.name, "HBM", before, after, "exit", completed.returncode, flush=True)
    if completed.returncode:
        raise RuntimeError(f"{variant} exited {completed.returncode} at {out}")
    return [before, after]


def wrapped(*args):
    return ["bash", "-c", 'set -eo pipefail; source ./env.sh; unset ASCEND_RT_VISIBLE_DEVICES; exec "$@"',
            "pta08", *args]


def gate(control, out, variant, stock):
    run_base = f"{INNER}/{out.parent.name}/{out.name}"
    invoke(control, out, variant, wrapped("torchrun", "--standalone", "--nnodes=1", "--nproc-per-node=1",
                                        f"{INNER}/validate_npu.py", "run", "--workspace", f"{INNER}/work",
                                        "--output", f"{run_base}/validation", "--capture-gradients", "--combined", "--pta"),
           stock=stock)


def correctness(control, out, variants):
    gate(control, out / "reference", "reference", False)
    passed = []
    for variant in variants:
        try:
            if variant in ("queued", "flat"):
                name = f"test-{variant}"
                invoke(control, out / name, variant, wrapped("python3", "-B", f"{INNER}/test_pta.py",
                       variant, f"{INNER}/{out.name}/{name}/result.json"))
            gate(control, out / variant, variant, False)
            a, b = (f"{INNER}/{out.name}/{v}/validation" for v in ("reference", variant))
            name = f"compare-{variant}"
            invoke(control, out / name, variant, wrapped("python3", "-B", f"{INNER}/compare_gate.py",
                   a, b, f"{INNER}/{out.name}/{name}/result.json"), verify_idle=False)
            passed.append(variant)
        except Exception as error:
            print("GATE_FAILED", variant, type(error).__name__, str(error), flush=True)
    write(out / "result.json", {"passed": passed, "requested": variants})


def trace_markers(rank, out, variant):
    (trace,) = rank.rglob("trace_view.json")
    document = read(trace)
    events = document["traceEvents"] if isinstance(document, dict) else document
    selected = {"NPUGraphCopyReplay", "NPUDenseFlattenStorageCopy",
                "aten::flatten_dense_tensors", "aten::_flatten_dense_tensors",
                "aten::unflatten_dense_tensors", "aten::_unflatten_dense_tensors"}
    results = {}
    for event in events:
        name = event.get("name", "")
        if event.get("ph") == "X" and (name in selected or name.startswith("ProfilerStep#")):
            results.setdefault(name, []).append({key: event[key] for key in ("ts", "dur")})
    flags = VARIANTS[variant]
    assert bool(results.get("NPUGraphCopyReplay")) is (flags[0] == "1"), sorted(results)
    assert bool(results.get("NPUDenseFlattenStorageCopy")) is (flags[1] == "1"), sorted(results)
    write(out / "trace-markers.json", results)


def performance(control, out, variant, mode):
    source = read(HERE / f"{mode}-command.json")
    argv = source["argv"]
    args = argv[argv.index("./run_training.sh") - 1:]
    args[args.index("--output-dir") + 1] = f"{INNER}/{out.relative_to(HERE).as_posix()}/training"
    hbm = invoke(control, out, variant, args)
    rank = out / "training" / "rank000"
    pta = read(rank / "pta_wrapper.json")
    assert pta["graph"] and pta["scope"] == "both"
    expected = 60 if mode == "steady" else 8
    assert pta["overflow"] == pta["norm"] == expected, pta
    assert read(rank / "hstu_graph.json")["counts"] == {"forward": expected, "backward": expected}
    summary, rows = training(rank / "metrics.json", "no-profiler" if mode == "steady" else "l0")
    if mode == "steady":
        evidence = steady_state(summary, rows)
    else:
        (csv,) = rank.rglob("step_trace_time.csv")
        evidence = profile(csv, "l0")
        shutil.copy2(csv, out / "step_trace_time.csv")
        trace_markers(rank, out, variant)
    return {"variant": variant, "hbm_percent": hbm, "values": evidence["values"],
            "window": evidence["window"], "sampled_loss": [row["loss"] for row in rows if "loss" in row]}


def measure(control, out, variants, mode, order_filter):
    pairs = []
    schedule = [("reference", "reference")] if mode == "steady" and order_filter == "both" else []
    for variant in variants:
        if order_filter in ("both", "ab"):
            schedule.append(("reference", variant))
        if order_filter in ("both", "ba"):
            schedule.append((variant, "reference"))
    for index, order in enumerate(schedule, 1):
        pair = out / f"{index:02d}-{'-'.join(order)}"
        pair.mkdir()
        try:
            legs = [performance(control, pair / f"{pos}-{variant}", variant, mode)
                    for pos, variant in enumerate(order, 1)]
            assert legs[0]["window"] == legs[1]["window"]
            assert all(math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-7)
                       for a, b in zip(legs[0]["sampled_loss"], legs[1]["sampled_loss"], strict=True))
            rates = [rate for leg in legs for rate in leg["hbm_percent"]]
            status = "matched" if max(rates) - min(rates) <= 2 else "variable_hbm"
            result = {"status": status, "order": order, "legs": legs}
        except Exception as error:
            result = {"status": "failed", "order": order, "error": f"{type(error).__name__}: {error}"}
        write(pair / "result.json", result)
        pairs.append(result)
        write(out / "result.json", pairs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("stock-audit", "gates", "steady", "level0"))
    parser.add_argument("name")
    parser.add_argument("variants", nargs="*")
    parser.add_argument("--package", default="package")
    parser.add_argument("--order", choices=("both", "ab", "ba"), default="both")
    args = parser.parse_args()
    assert "/" not in args.name and args.name not in ("", ".", "..")
    assert all(variant in ("queued", "flat", "combined") for variant in args.variants)
    out = HERE / args.name
    out.mkdir()
    with LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        container, init_pid = inspect_container(ARTIFACTS, out)
        control = container, init_pid, args.package
        if args.stage == "stock-audit":
            gate(control, out / "reference", "reference", True)
        elif args.stage == "gates":
            correctness(control, out, args.variants)
        else:
            measure(control, out, args.variants, args.stage, args.order)


if __name__ == "__main__":
    main()
