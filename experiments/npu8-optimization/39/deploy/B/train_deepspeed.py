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
    torch.manual_seed(args.seed)
    model = MultiTableHSTU(config)
    summary = model_summary(model, config)
    if args.dry_run:
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
    if local_rank != 0 or os.environ.get("WORLD_SIZE") != "1" or torch.npu.device_count() != 1:
        raise RuntimeError("26 integration requires exactly one reserved, container-local NPU")
    if deepspeed.__version__ != "0.19.0" or args.save_checkpoint:
        raise RuntimeError("26 integration requires DeepSpeed 0.19.0 and disables DS checkpoint save")
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
    overflow_stats = None
    overflow_switch = os.environ.get("RECSYS_ZERO2_BATCH_OVERFLOW", "0")
    if overflow_switch != "0":
        from overflow import install

        overflow_stats = install(engine.optimizer)
    owner26 = None
    finish_graph = None
    fast_partition = None
    norm_stats = None
    finish_pta = None
    if os.environ.get("RECSYS_HSTU_COMBINED") == "1":
        from graphed_hstu import install as install_graph, install_single_rank_partition
        from zero2_batch_norm import finish_run as finish_norm_run, install as install_norm

        os.environ["RECSYS_ZERO2_BATCH_NORM"] = "1"
        norm_stats = install_norm(engine.optimizer)
        from owner26 import ExecutorGradient, attach
        finish_graph = install_graph(engine, config, args.micro_batch_size, device, rank_dir,
                                     direct_factory=ExecutorGradient)
        fast_partition = install_single_rank_partition(engine, device)
        if os.environ.get("RECSYS_PTA_WRAPPER") == "1":
            from pta_wrapper import install as install_pta
            owner26, finish_pta = attach(engine, device, rank_dir, norm_stats, install_pta)

    (rank_dir / "model_summary.json").write_text(json.dumps(summary, indent=2))
    (rank_dir / "effective_deepspeed_config.json").write_text(json.dumps(ds_config, indent=2))
    (rank_dir / "run_arguments.json").write_text(json.dumps(vars(args), indent=2))
    if rank == 0:
        print(json.dumps({"model": summary, "world_size": world_size, "profile": args.profile}, indent=2), flush=True)

    # Keep data generation on CPU so NPU RNG support is not required. Each table
    # participates in the model; no disconnected/dummy parameters are added.
    generator = torch.Generator(device="cpu").manual_seed(args.seed + rank)
    overlap_mode = os.environ.get("RECSYS_OVERLAP_INPUT", "0")
    audit_mode = os.environ.get("RECSYS_OVERLAP_INPUT_AUDIT", "0")
    if overlap_mode not in ("0", "1") or audit_mode not in ("0", "1"):
        raise ValueError("input overlap and audit switches must be 0 or 1")
    overlap_input = None
    input_audit = [] if audit_mode == "1" else None
    device_input_audit = [] if input_audit is not None else None
    if input_audit is not None:
        from overlap_input import tensor_digest
    if overlap_mode == "1":
        direct = getattr(engine.optimizer, "_recsys_direct_gradient", None)
        if (world_size != 1 or accumulation != 1 or direct is None or not direct.epilogue_installed
                or direct.stream != torch.npu.current_stream(device)):
            raise RuntimeError("input overlap requires the original single-rank 13 training stream")
        from overlap_input import OverlapInput

        overlap_input = OverlapInput(torch.npu, device, config, args.micro_batch_size, generator)
    if None in (finish_graph, finish_pta, overlap_input) or args.micro_batch_size != 2:
        raise RuntimeError("26 requires Train Replay, PTA and overlap input with micro-batch 2")
    audit26 = None
    if os.environ.get("RECSYS_26_INTEGRATION_MODE") == "correctness":
        from audit26 import Audit26
        audit26 = Audit26(engine, owner26, rank_dir, args.steps)
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
    drained = False
    try:
        with profiler_context as profiler:
            for step in range(args.steps):
                if steady and step == args.steady_warmup_steps:
                    torch.npu.synchronize()
                    steady_start = time.perf_counter()
                step_start = time.perf_counter()
                with record_function("input_generation_and_h2d"):
                    inputs = overlap_input.take(step) if overlap_input is not None else None
                    if inputs is None:
                        tokens = torch.randint(1, config.vocab_size,
                                               (args.micro_batch_size, config.seq_len + 1), generator=generator)
                        cpu_items = tokens[:, :-1].contiguous()
                        items = cpu_items.to(device)
                        cpu_targets = tokens[:, 1:].contiguous()
                        targets = cpu_targets.to(device)
                        cpu_groups = torch.randint(0, config.num_user_groups,
                                                   (args.micro_batch_size,), generator=generator)
                        groups = cpu_groups.to(device)
                        cpu_features = torch.randint(0, config.table_rows,
                                                     (args.micro_batch_size, config.seq_len,
                                                      config.num_tables, config.features_per_table),
                                                     generator=generator)
                        features = cpu_features.to(device)
                        if input_audit is not None:
                            input_audit.append(tensor_digest((cpu_items, cpu_targets, cpu_groups, cpu_features)))
                        del cpu_items, cpu_targets, cpu_groups, cpu_features
                    else:
                        items, targets, groups, features = inputs
                    if device_input_audit is not None:
                        device_input_audit.append(tensor_digest(tuple(
                            value.detach().cpu() for value in (items, targets, groups, features))))
                with record_function("hstu_forward_and_loss"):
                    logits = engine(items, groups, features)
                    loss = functional.cross_entropy(logits.float().reshape(-1, config.vocab_size),
                                                    targets.reshape(-1))
                with record_function("deepspeed_backward"):
                    engine.backward(loss)
                if overlap_input is not None:
                    with record_function("input_overlap_after_backward"):
                        overlap_input.release()
                        next_step = step + 1
                        # Keep the first measured batch on the timed side of the warmup fence.
                        if next_step < args.steps and not (steady and next_step == args.steady_warmup_steps):
                            overlap_input.submit(next_step, input_audit)
                if audit26 is not None:
                    audit26.before(step, loss)
                with record_function("optimizer_step_including_native_deepspeed_norm"):
                    engine.step()
                if audit26 is not None:
                    audit26.after()
                metric = {"step": step, "host_step_ms": (time.perf_counter() - step_start) * 1000}
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
        drained = True
        elapsed = end - start
        result = {"status": "completed", "rank": rank, "steps": args.steps,
                  "elapsed_s_synchronized": elapsed, "profile_enabled": args.profile,
                  "samples_per_s": args.steps * args.micro_batch_size * world_size / elapsed,
                  "profiled_elapsed_includes_profiler_overhead": args.profile}
        result["overlap_input"] = overlap_input is not None
        result["overlap_input_hits"] = overlap_input.consumed if overlap_input is not None else 0
        if overlap_input is not None:
            overlap_input.verify_finished(args.steps - 1 - int(steady))
        if input_audit is not None:
            if len(input_audit) != args.steps:
                raise RuntimeError("input audit did not record every batch")
            if device_input_audit != input_audit:
                raise RuntimeError("device input differs from the CPU generator output")
            (rank_dir / "overlap_input_audit.json").write_text(json.dumps({
                "input_sha256": input_audit, "device_input_sha256": device_input_audit,
                "final_rng_sha256": tensor_digest((generator.get_state(),)),
            }, indent=2) + "\n", encoding="utf-8")
        if steady:
            steady_elapsed = end - steady_start
            samples = args.steady_measure_steps * args.micro_batch_size * world_size
            result["steady_state"] = {
                "start_step_inclusive": args.steady_warmup_steps,
                "end_step_exclusive": args.steps,
                "steps": args.steady_measure_steps,
                "samples": samples,
                "elapsed_s_synchronized": steady_elapsed,
                "samples_per_s": samples / steady_elapsed,
                "ms_per_step": steady_elapsed * 1000 / args.steady_measure_steps,
                "loss_log_policy": "deferred_until_after_end_sync",
            }
            for metric, logged_loss in deferred_logs:
                metric["loss"] = logged_loss.float().cpu().item()
                print(json.dumps(metric), flush=True)
        if finish_graph is not None:
            finish_graph(args.steps)
            if owner26 is None:
                if fast_partition["partition"] != args.steps:
                    raise RuntimeError(f"unexpected partition count: {fast_partition}")
                (rank_dir / "fast_partition.json").write_text(json.dumps(fast_partition))
            else:
                owner26.finish(args.steps, rank_dir)
            updates = args.steps - engine.skipped_steps
            if audit26 is None and updates != args.steps:
                raise RuntimeError("timing leg saw overflow and did unequal work")
            finish_norm_run(norm_stats, rank_dir, updates, True)
            if finish_pta is not None:
                finish_pta()
        if audit26 is not None:
            audit26.finish()
        (rank_dir / "metrics.json").write_text(json.dumps({"summary": result, "steps": metrics}, indent=2))
        if args.save_checkpoint:
            engine.save_checkpoint(str(output_dir / "checkpoint"), tag="final")
        print(json.dumps(result), flush=True)
    finally:
        try:
            if overlap_input is not None and not drained:
                overlap_input.drain_after_failure()
            if overflow_stats is not None:
                from overflow import finish_run

                finish_run(overflow_stats, rank_dir, args.steps, sys.exc_info()[0] is None)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    main()
