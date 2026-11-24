"""Explicit single-NPU ZeRO-2 semantic check, and an offline A/B report comparison."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys


CASES = ("finite_below_clip", "nan", "posinf", "neginf", "finite_recovery_above_clip")


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run(args):
    require("LOCAL_RANK" in os.environ and os.environ.get("WORLD_SIZE") == "1",
            "run only under an explicitly reserved one-rank torchrun")
    require(os.environ.get("RECSYS_NPU_IDS", "").isdigit(), "one selected container NPU is required")
    os.environ["DS_ACCELERATOR"] = "npu"
    sys.path.insert(0, str(args.workspace.resolve(strict=True)))
    if args.graph_sidepack:
        sys.path.insert(0, str(Path(__file__).resolve().parent))

    import torch
    import torch_npu
    import deepspeed
    from model import ModelConfig, MultiTableHSTU
    from overflow import ACL_FORMAT_ND, CHUNK_ELEMENTS, enabled, finish_run, install, shard_metadata

    candidate = enabled()
    require(not args.pta or candidate, "PTA validation requires the candidate overflow path")
    args.output.mkdir(parents=True, exist_ok=False)
    report_path = args.output / "correctness.json"
    report = {"variant": "B" if candidate else "A", "status": "running", "steps": []}
    if args.capture_gradients:
        report["capture_gradients"] = True
        report["graph"] = args.graph
    if args.combined:
        report["combined"] = True
    if args.pta:
        report["pta"] = True
        if os.environ.get("RECSYS_PTA_BOUND_STATS") == "1":
            report["bound_stats"] = True
        if os.environ.get("RECSYS_PTA_LOCAL_STATISTICS") == "1":
            report["rank_one_stats"] = True
        local_step = os.environ.get("RECSYS_ZERO2_LOCAL_STEP")
        if local_step:
            report["zero2_local_step"] = local_step
    if args.graph_sidepack:
        report["graph_sidepack"] = True
        report["direct_gradient"] = os.environ.get("RECSYS_HSTU_DIRECT_GRADIENT") == "1"
    write_report(report_path, report)
    rank = int(os.environ["LOCAL_RANK"])
    stats = None
    norm_events, partition_counts, finish_pta = None, None, None

    def digest(tensor):
        cpu = tensor.detach().cpu().contiguous().view(-1).view(torch.uint8)
        return {
            "dtype": str(tensor.dtype), "shape": list(tensor.shape),
            "sha256": hashlib.sha256(cpu.numpy().tobytes()).hexdigest(),
        }

    def snapshot(zero, adamw):
        masters = zero.single_partition_of_fp32_groups
        state = []
        for param in masters:
            values = adamw.state.get(param, {})
            state.append({key: digest(value) if torch.is_tensor(value) else value
                          for key, value in sorted(values.items())})
        return {"master": [digest(tensor) for tensor in masters],
                "bf16": [digest(tensor) for tensor in zero.bit16_groups_flat],
                "adamw": state,
                "adam_steps": [int(adamw.state.get(param, {}).get("step", 0)) for param in masters]}

    try:
        torch.npu.set_device(rank)
        device = torch.device("npu", rank)
        deepspeed.init_distributed(dist_backend="hccl", auto_mpi_discovery=False)
        torch.manual_seed(args.seed)
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        workspace = args.workspace.resolve()
        limits = json.loads((workspace / "configs/model_many_tables.json").read_text())
        model_config = ModelConfig(**limits)
        model = MultiTableHSTU(model_config)
        ds_config = json.loads((workspace / "configs/deepspeed_zero2_bf16_overflow_on.json").read_text())
        ds_config.update(train_micro_batch_size_per_gpu=2, train_batch_size=2)
        require(ds_config["gradient_accumulation_steps"] == 1 and ds_config["gradient_clipping"] == 1.0,
                "validator requires one micro-step per update and clip_grad=1")
        adamw = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01, foreach=False)
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=adamw, config=ds_config,
                                              dist_init_required=False)
        engine.train()
        zero = engine.optimizer
        original_scan = zero.has_overflow_partitioned_grads_serial
        if candidate:
            stats = install(zero)
        else:
            require(zero.has_overflow_partitioned_grads_serial == original_scan
                    and "_recsys_overflow_stats" not in vars(zero), "A must retain original scan")
        finish_graph = None
        if args.graph or args.combined:
            from graphed_hstu import install as install_graph

            finish_graph = install_graph(engine, model_config, 2, device, args.output)
        if args.combined:
            from graphed_hstu import install_single_rank_partition
            from zero2_batch_norm import install as install_norm

            partition_counts = install_single_rank_partition(engine, device)
            native_norm = zero.get_grad_norm_direct
            norm_stats = install_norm(zero)
            require(norm_stats is not None, "combined requires enabled batched norm")
            if args.pta:
                from pta_wrapper import install as install_pta

                finish_pta = install_pta(engine, args.output, norm_stats)
            batched_norm = zero.get_grad_norm_direct
            norm_events = []

            def check_norm(gradients, params, norm_type=2):
                original_value = native_norm(gradients, params, norm_type)
                candidate_value = batched_norm(gradients, params, norm_type)
                torch.testing.assert_close(candidate_value.cpu(), original_value.cpu(), rtol=1e-5, atol=1e-7)
                norm_events.append({"native": original_value.item(), "batched": candidate_value.item()})
                return candidate_value

            zero.get_grad_norm_direct = check_norm
        report["runtime"] = {"python": sys.version.split()[0], "torch": torch.__version__,
                             "torch_npu": torch_npu.__version__, "deepspeed": deepspeed.__version__,
                             "device": str(device), "seed": args.seed,
                             "grad_accum_dtype": str(zero.gradient_accumulation_dtype),
                             "chunk_elements": CHUNK_ELEMENTS}
        report["effective_config"] = ds_config
        report["model_config"] = limits
        if args.capture_gradients:
            names = {id(parameter): name for name, parameter in engine.module.named_parameters()}
            report["shard_order"] = [[names[id(parameter)] for parameter in group]
                                     for group in zero.params_in_partition]
        write_report(report_path, report)

        original_clip = zero.unscale_and_clip_grads
        clip_events = []

        def record_clip(grad_groups, scaled_norm):
            original_clip(grad_groups, scaled_norm)
            norm = float(scaled_norm.item())
            scale = float(zero.loss_scale)
            clip_events.append({"scaled_norm": norm, "loss_scale": scale,
                                "reference_clip_factor": 1.0 / (scale * max(1.0, (norm / scale + 1e-6) / zero.clip_grad)),
                                "clipped_fp32": [digest(grad) for grad in grad_groups]})

        zero.unscale_and_clip_grads = record_clip
        for index, case in enumerate(CASES):
            clip_events.clear()
            tokens = torch.randint(1, model_config.vocab_size, (2, model_config.seq_len + 1), generator=generator)
            items, targets = (tokens[:, :-1].contiguous().to(device), tokens[:, 1:].contiguous().to(device))
            users = torch.randint(0, model_config.num_user_groups, (2,), generator=generator).to(device)
            features = torch.randint(
                0, model_config.table_rows,
                (2, model_config.seq_len, model_config.num_tables, model_config.features_per_table),
                generator=generator).to(device)
            logits = engine(items, users, features)
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, model_config.vocab_size),
                                                     targets.reshape(-1))
            require(math.isfinite(loss.item()), "expected finite HSTU loss before gradient injection")
            engine.backward(loss)
            groups = [zero.averaged_gradients[group] for group in range(len(zero.bit16_groups))]
            grads = [grad for group in groups for grad in group if grad is not None]
            require(len(grads) > 1 and all(grad.dtype == torch.bfloat16 and grad.device == device
                    and grad.numel() > 0 and grad.is_contiguous()
                    and torch.ops.npu.get_npu_format(grad) == ACL_FORMAT_ND for grad in grads),
                    "expected real contiguous ND BF16 ZeRO-2 partitions")
            if args.capture_gradients:
                count = len(tuple(engine.module.parameters()))
                require(len(grads) in (count, count + 1), "unexpected padding count")
                torch.save([grad.detach().cpu().clone() for grad in grads], args.output / f"gradients-{index}.pt")
            invalid = case in ("nan", "posinf", "neginf")
            with torch.no_grad():
                for grad in grads:
                    grad.fill_(2 ** -14)
                if invalid:
                    target = grads[0] if case == "posinf" else grads[-1]
                    sentinel = {"nan": float("nan"), "posinf": float("inf"), "neginf": -float("inf")}
                    target.view(-1)[-1].fill_(sentinel[case])
                elif case == CASES[-1]:
                    grads[-1].view(-1)[-1].fill_(16.0)

            skipped_before = engine.skipped_steps
            before = snapshot(zero, adamw)
            fast_before = stats.fast_calls if stats is not None else None
            step = {"index": index, "case": case, "loss": float(loss.item()),
                    "shards": shard_metadata(groups), "before": before}
            report["steps"].append(step)
            write_report(report_path, report)
            engine.step()
            torch.npu.synchronize()
            applied = engine.was_step_applied()
            after = snapshot(zero, adamw)
            skip_delta = engine.skipped_steps - skipped_before
            step.update(overflow=bool(zero.overflow), applied=applied, skip_delta=skip_delta,
                        clip=list(clip_events), after=after,
                        cleared=all(grad is None for grad in zero.averaged_gradients.values())
                        and all(grad is None for grad in zero.all_grad_tensors.values())
                        and all(param.grad is None for param in engine.module.parameters()))
            write_report(report_path, report)
            require(step["cleared"], f"{case}: gradients not cleared")
            require(step["overflow"] == invalid and applied != invalid and skip_delta == int(invalid),
                    f"{case}: incorrect overflow or update decision")
            if stats is not None:
                require(stats.fast_calls == fast_before + 1 and stats.fallback_calls == stats.errors == 0,
                        f"{case}: candidate did not hit fast path exactly once")
            if invalid:
                require(not clip_events and before == after, f"{case}: overflow changed parameters/AdamW or clipped")
            else:
                require(len(clip_events) == len(zero.bit16_groups)
                        and after["master"] != before["master"]
                        and after["adam_steps"] == [value + 1 for value in before["adam_steps"]]
                        and all({"exp_avg", "exp_avg_sq"} <= values.keys() for values in after["adamw"]),
                        f"{case}: missing AdamW update")
                require(all((event["reference_clip_factor"] < 1) == (case == CASES[-1])
                            for event in clip_events), f"{case}: wrong side of clip threshold")

        if stats is not None:
            finish_run(stats, args.output, len(CASES), True)
        if finish_graph is not None:
            finish_graph(len(CASES))
        if finish_pta is not None:
            finish_pta()
        if args.combined:
            require(partition_counts["partition"] == len(CASES) and len(norm_events) == 2,
                    "missing batched partition or norm evidence")
            norm_stats.require_batched_step(2)
            report["partition"] = partition_counts
            report["norm_events"] = norm_events
            report["norm_stats"] = norm_stats.snapshot()
        report["status"] = "passed"
        write_report(report_path, report)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        if stats is not None:
            report["scan"] = stats.snapshot()
        write_report(report_path, report)
        raise
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def compare(args):
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = {"status": "failed", "baseline": str(args.baseline), "candidate": str(args.candidate)}
    try:
        require(baseline["variant"] == "A" and candidate["variant"] == "B", "expected A and B in that order")
        require(baseline["status"] == candidate["status"] == "passed", "both correctness runs must pass")
        require(not (args.baseline.parent / "overflow_scan.json").exists(), "A emitted candidate scan evidence")
        scan = json.loads((args.candidate.parent / "overflow_scan.json").read_text(encoding="utf-8"))
        require(scan["valid"] and scan["calls"] == scan["fast_calls"] == len(CASES)
                and scan["fallback_calls"] == scan["errors"] == 0, "B scan evidence invalid")
        require(scan["first_npu_formats"] == {"2": sum(entry["count"] for entry in scan["first_shards"]
                                                     if entry["dtype"] == "torch.bfloat16")},
                "B did not report actual ND BF16 shards")
        for field in ("runtime", "effective_config", "model_config"):
            require(baseline[field] == candidate[field], f"A/B differs in {field}")
        fields = ("case", "shards", "overflow", "applied", "skip_delta", "cleared", "clip", "before", "after")
        require(tuple(row["case"] for row in baseline["steps"]) == CASES, "wrong A case sequence")
        require(tuple(row["case"] for row in candidate["steps"]) == CASES, "wrong B case sequence")
        for left, right in zip(baseline["steps"], candidate["steps"], strict=True):
            require(math.isclose(left["loss"], right["loss"], rel_tol=1e-6, abs_tol=1e-7),
                    f"{left['case']}: forward loss differs")
            for field in fields:
                require(left[field] == right[field], f"{left['case']}: A/B differs in {field}")
        report.update(status="passed", cases=list(CASES), compared=list(fields) + ["loss"], scan=scan)
    except Exception as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        write_report(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    npu = commands.add_parser("run")
    npu.add_argument("--workspace", type=Path, required=True)
    npu.add_argument("--output", type=Path, required=True)
    npu.add_argument("--seed", type=int, default=1234)
    npu.add_argument("--graph", action="store_true")
    npu.add_argument("--capture-gradients", action="store_true")
    npu.add_argument("--combined", action="store_true")
    npu.add_argument("--pta", action="store_true")
    npu.add_argument("--graph-sidepack", action="store_true")
    npu.set_defaults(action=run)
    offline = commands.add_parser("compare")
    offline.add_argument("baseline", type=Path)
    offline.add_argument("candidate", type=Path)
    offline.add_argument("--output", type=Path, required=True)
    offline.set_defaults(action=compare)
    args = parser.parse_args()
    if getattr(args, "pta", False) and not args.combined:
        parser.error("--pta requires --combined")
    if getattr(args, "graph_sidepack", False) and not args.pta:
        parser.error("--graph-sidepack requires --pta")
    args.action(args)


if __name__ == "__main__":
    main()
