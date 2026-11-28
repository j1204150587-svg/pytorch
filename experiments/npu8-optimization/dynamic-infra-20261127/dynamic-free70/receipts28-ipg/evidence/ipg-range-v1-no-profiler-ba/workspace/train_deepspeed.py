"""Prepared multi-table HSTU workload; execution is explicit, never automatic."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import time


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", default="configs/model_default.json")
    parser.add_argument("--deepspeed-config", default="configs/deepspeed_zero2_bf16.json")
    parser.add_argument("--dry-run", action="store_true", help="CPU construction/parameter count only; no forward, DeepSpeed or NPU")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--shape-schedule", type=Path,
                        help="JSON array of [batch size, sequence length] pairs, one pair per step")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--steady-warmup-steps", type=int, default=0)
    parser.add_argument("--steady-measure-steps", type=int, default=0)
    parser.add_argument("--output-dir", default="/artifacts/hstu_deepspeed")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-active", type=int, default=5)
    parser.add_argument("--profile-stack", action="store_true")
    parser.add_argument("--profile-level1", action="store_true", help="collect CANN runtime dispatch events")
    parser.add_argument("--save-checkpoint", action="store_true")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=-1)
    for key in ("num_tables", "table_rows", "features_per_table", "seq_len", "embedding_dim", "num_layers"):
        parser.add_argument("--" + key.replace("_", "-"), type=int)
    args = parser.parse_args()
    if min(args.steps, args.micro_batch_size, args.log_interval) < 1:
        parser.error("steps, micro-batch-size and log-interval must be positive")
    if args.profile and (args.profile_active < 1 or args.steps < 4 + args.profile_active):
        parser.error("profile requires 2 wait + 2 warmup steps and at least one active step")
    warmup, measured = args.steady_warmup_steps, args.steady_measure_steps
    if warmup or measured:
        if min(warmup, measured) < 1 or warmup + measured != args.steps:
            parser.error("steady timing requires positive warmup and measure counts summing to --steps")
        if args.profile or args.dry_run:
            parser.error("steady timing requires training with profiler disabled")
    return args


def input_shapes(args, max_seq_len):
    if args.shape_schedule is None:
        return [(args.micro_batch_size, max_seq_len)] * args.steps
    schedule = json.loads(args.shape_schedule.read_text(encoding="utf-8"))
    if not isinstance(schedule, list) or len(schedule) < args.steps:
        raise ValueError("shape schedule must contain at least --steps pairs")
    shapes = []
    for step, pair in enumerate(schedule[:args.steps]):
        if (not isinstance(pair, list) or len(pair) != 2 or any(type(value) is not int for value in pair)
                or pair[0] < 1 or not 1 <= pair[1] <= max_seq_len):
            raise ValueError(f"shape schedule step {step} requires [batch >= 1, 1 <= length <= {max_seq_len}]")
        shapes.append(tuple(pair))
    return shapes


def main():
    args = parse_args()
    if args.dry_run:
        os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["DS_ACCELERATOR"] = "npu"
    import torch
    from model import ModelConfig, MultiTableHSTU, model_summary

    values = json.loads(Path(args.model_config).read_text())
    for key in ("num_tables", "table_rows", "features_per_table", "seq_len", "embedding_dim", "num_layers"):
        if getattr(args, key) is not None:
            values[key] = getattr(args, key)
    config = ModelConfig(**values)
    shapes = input_shapes(args, config.seq_len)
    torch.manual_seed(args.seed)
    model = MultiTableHSTU(config)
    summary = model_summary(model, config)
    if args.dry_run:
        summary["input_shapes"] = sorted(set(shapes))
        assert all(parameter.device.type == "cpu" for parameter in model.parameters())
        assert "deepspeed" not in sys.modules and "torch_npu" not in sys.modules
        summary["verification"] = "CPU construction only; no forward/backward, DeepSpeed import, NPU initialization or profile"
        print(json.dumps(summary, indent=2))
        return

    # No hardware code is reached by --help or --dry-run.
    import torch_npu
    import deepspeed
    import torch.distributed as dist
    import torch.nn.functional as functional
    from torch.profiler import record_function

    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    if local_rank < 0 or "RANK" not in os.environ:
        raise RuntimeError("Launch with run_training.sh / torchrun, after reserving NPU devices")
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    deepspeed.init_distributed(dist_backend="hccl", auto_mpi_discovery=False)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    if args.steady_measure_steps and world_size != 1:
        raise RuntimeError("steady timing requires a single rank")
    output_dir = Path(args.output_dir)
    rank_dir = output_dir / f"rank{rank:03d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
    (rank_dir / "runtime.json").write_text(json.dumps({
        "rank": rank, "world_size": world_size, "local_rank": local_rank,
        "device": str(device), "python": sys.version.split()[0],
        "torch": torch.__version__, "torch_npu": torch_npu.__version__,
        "deepspeed": deepspeed.__version__, "deepspeed_file": deepspeed.__file__,
    }, indent=2))
    ds_config = json.loads(Path(args.deepspeed_config).read_text())
    ds_config["train_micro_batch_size_per_gpu"] = args.micro_batch_size
    accumulation = int(ds_config.get("gradient_accumulation_steps", 1))
    if args.shape_schedule is not None and (world_size != 1 or accumulation != 1):
        raise RuntimeError("shape schedule requires one rank and gradient accumulation 1")
    ds_config["train_batch_size"] = args.micro_batch_size * accumulation * world_size
    assert ds_config["zero_optimization"]["stage"] == 2
    assert ds_config["bf16"]["enabled"] is True
    assert ds_config["gradient_clipping"] > 0
    assert not ds_config["zero_optimization"].get("offload_optimizer")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                 weight_decay=args.weight_decay, foreach=False)
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer,
                                          config=ds_config, dist_init_required=False)
    engine.train()
    from range_copy import install as install_ipg_range
    ipg_range=install_ipg_range(engine)
    overflow_stats = None
    overflow_switch = os.environ.get("RECSYS_ZERO2_BATCH_OVERFLOW", "0")
    if overflow_switch != "0":
        from overflow import install

        overflow_stats = install(engine.optimizer)
    partition_enabled = any(os.environ.get(key) == "1" for key in (
        "RECSYS_DYNAMIC_PARTITION", "RECSYS_DYNAMIC_BOTH"))
    norm_enabled = any(os.environ.get(key) == "1" for key in (
        "RECSYS_DYNAMIC_NORM", "RECSYS_DYNAMIC_BOTH"))
    audit = os.environ.get("RECSYS_DYNAMIC_STATE_AUDIT") == "1"
    partition_stats = norm_stats = None
    if partition_enabled:
        from graphed_hstu import install_single_rank_partition
        partition_stats = install_single_rank_partition(engine, device)
    if norm_enabled:
        from zero2_batch_norm import finish_run as finish_norm_run, install as install_norm
        os.environ["RECSYS_ZERO2_BATCH_NORM"] = "1"
        norm_stats = install_norm(engine.optimizer)
    if audit:
        import hashlib

        def digest(tensor):
            values = tensor.detach().reshape(-1).contiguous().cpu().view(torch.uint8)
            return hashlib.sha256(values.numpy().tobytes()).hexdigest()

        audits = []
        get_norm = engine.optimizer.get_grad_norm_direct

        def audit_norm(gradients, parameters, norm_type=2):
            flat = torch.cat([gradient.detach().reshape(-1) for gradient in gradients])
            norm = get_norm(gradients, parameters, norm_type)
            audits.append({"gradients": digest(flat), "norm": norm.detach().item()})
            return norm

        engine.optimizer.get_grad_norm_direct = audit_norm
    (rank_dir / "model_summary.json").write_text(json.dumps(summary, indent=2))
    (rank_dir / "effective_deepspeed_config.json").write_text(json.dumps(ds_config, indent=2))
    (rank_dir / "run_arguments.json").write_text(json.dumps(vars(args), indent=2, default=str))
    if rank == 0:
        print(json.dumps({"model": summary, "world_size": world_size, "profile": args.profile}, indent=2), flush=True)

    # Keep data generation on CPU so NPU RNG support is not required. Each table
    # participates in the model; no disconnected/dummy parameters are added.
    generator = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    profiler_context = nullcontext(None)
    if args.profile:
        # One host annotation per native ZeRO norm pass. Keep the installed implementation.
        native_norm = engine.optimizer.get_grad_norm_direct
        def traced_native_norm(*grad_args, **grad_kwargs):
            with record_function("deepspeed_native_get_grad_norm_direct"):
                return native_norm(*grad_args, **grad_kwargs)
        engine.optimizer.get_grad_norm_direct = traced_native_norm
        experimental = {}
        if args.profile_level1:
            experimental["experimental_config"] = torch_npu.profiler._ExperimentalConfig(
                profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                aic_metrics=torch_npu.profiler.AiCMetrics.AiCoreNone)
        profiler_context = torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU],
            schedule=torch_npu.profiler.schedule(wait=2, warmup=2, active=args.profile_active, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(rank_dir / "profile"), analyse_flag=True),
            record_shapes=False, with_stack=args.profile_stack, profile_memory=False, **experimental)
    metrics = []
    steady = args.steady_measure_steps > 0
    deferred_logs = []
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    try:
        with profiler_context as profiler:
            for step, (batch_size, sequence_length) in enumerate(shapes):
                if steady and step == args.steady_warmup_steps:
                    torch.npu.synchronize()
                    steady_start = time.perf_counter()
                step_start = time.perf_counter()
                with record_function("input_generation_and_h2d"):
                    tokens = torch.randint(1, config.vocab_size,
                                           (batch_size, sequence_length + 1), generator=generator)
                    items = tokens[:, :-1].contiguous().to(device)
                    targets = tokens[:, 1:].contiguous().to(device)
                    groups = torch.randint(0, config.num_user_groups,
                                           (batch_size,), generator=generator).to(device)
                    features = torch.randint(0, config.table_rows,
                                             (batch_size, sequence_length,
                                              config.num_tables, config.features_per_table),
                                             generator=generator).to(device)
                with record_function("hstu_forward_and_loss"):
                    logits = engine(items, groups, features)
                    loss = functional.cross_entropy(logits.float().reshape(-1, config.vocab_size),
                                                    targets.reshape(-1))
                with record_function("deepspeed_backward"):
                    engine.backward(loss)
                with record_function("optimizer_step_including_native_deepspeed_norm"):
                    engine.step()
                metric = {"step": step, "batch_size": batch_size, "seq_len": sequence_length,
                          "host_step_ms": (time.perf_counter() - step_start) * 1000}
                # item() synchronizes only at log points; host_step_ms is not NPU kernel time.
                if step == 0 or (step + 1) % args.log_interval == 0:
                    if steady:
                        # Keep just the detached scalar. A device-to-host copy would
                        # serialize outstanding work inside the measurement window.
                        deferred_logs.append((metric, loss.detach()))
                    else:
                        metric["loss"] = loss.detach().float().cpu().item()
                        if rank == 0:
                            print(json.dumps(metric), flush=True)
                metrics.append(metric)
                if profiler is not None:
                    profiler.step()
        torch.npu.synchronize()
        end = time.perf_counter()
        elapsed = end - start
        samples = sum(batch for batch, _ in shapes) * world_size
        tokens = sum(batch * length for batch, length in shapes) * world_size
        result = {"status": "completed", "rank": rank, "steps": args.steps,
                  "elapsed_s_synchronized": elapsed, "profile_enabled": args.profile,
                  "input_shape_count": len(set(shapes)), "samples": samples, "tokens": tokens,
                  "samples_per_s": samples / elapsed, "tokens_per_s": tokens / elapsed,
                  "profiled_elapsed_includes_profiler_overhead": args.profile}
        if steady:
            steady_elapsed = end - steady_start
            measured_shapes = shapes[args.steady_warmup_steps:]
            samples = sum(batch for batch, _ in measured_shapes) * world_size
            tokens = sum(batch * length for batch, length in measured_shapes) * world_size
            result["steady_state"] = {
                "start_step_inclusive": args.steady_warmup_steps,
                "end_step_exclusive": args.steps,
                "steps": args.steady_measure_steps,
                "samples": samples,
                "tokens": tokens,
                "elapsed_s_synchronized": steady_elapsed,
                "samples_per_s": samples / steady_elapsed,
                "tokens_per_s": tokens / steady_elapsed,
                "ms_per_step": steady_elapsed * 1000 / args.steady_measure_steps,
                "loss_log_policy": "deferred_until_after_end_sync",
            }
            for metric, logged_loss in deferred_logs:
                metric["loss"] = logged_loss.float().cpu().item()
                print(json.dumps(metric), flush=True)
        if partition_enabled:
            if partition_stats["partition"] != args.steps:
                raise RuntimeError(f"unexpected number of local partitions: {partition_stats}")
            (rank_dir / "fast_partition.json").write_text(json.dumps(partition_stats))
        if norm_enabled:
            finish_norm_run(norm_stats, rank_dir, args.steps, True)
        if audit:
            masters = engine.optimizer.single_partition_of_fp32_groups
            states = [engine.optimizer.optimizer.state[master] for master in masters]
            state = {"gradients_and_norms": audits,
                     "bf16_weights": digest(torch.cat([p.detach().reshape(-1) for p in model.parameters()])),
                     "fp32_masters": [digest(p) for p in masters],
                     "adamw_exp_avg": [digest(s["exp_avg"]) for s in states],
                     "adamw_exp_avg_sq": [digest(s["exp_avg_sq"]) for s in states],
                     "adamw_steps": [int(s["step"].item()) for s in states]}
            (rank_dir / "state_audit.json").write_text(json.dumps(state, indent=2))
        (rank_dir / "metrics.json").write_text(json.dumps({"summary": result, "steps": metrics}, indent=2))
        ipg_range.finish(rank_dir)
        if args.save_checkpoint:
            engine.save_checkpoint(str(output_dir / "checkpoint"), tag="final")
        print(json.dumps(result), flush=True)
    finally:
        try:
            if overflow_stats is not None:
                from overflow import finish_run

                finish_run(overflow_stats, rank_dir, args.steps, sys.exc_info()[0] is None)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
