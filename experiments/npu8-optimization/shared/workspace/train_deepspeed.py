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
    torch.npu.set_device(local_rank)
    device = torch.device("npu", local_rank)
    deepspeed.init_distributed(dist_backend="hccl", auto_mpi_discovery=False)
    rank, world_size = dist.get_rank(), dist.get_world_size()
    output_dir = Path(args.output_dir)
    rank_dir = output_dir / f"rank{rank:03d}"
    rank_dir.mkdir(parents=True, exist_ok=True)
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
    (rank_dir / "model_summary.json").write_text(json.dumps(summary, indent=2))
    (rank_dir / "effective_deepspeed_config.json").write_text(json.dumps(ds_config, indent=2))
    (rank_dir / "run_arguments.json").write_text(json.dumps(vars(args), indent=2))
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
    dist.barrier()
    torch.npu.synchronize()
    start = time.perf_counter()
    try:
        with profiler_context as profiler:
            for step in range(args.steps):
                step_start = time.perf_counter()
                with record_function("input_generation_and_h2d"):
                    tokens = torch.randint(1, config.vocab_size,
                                           (args.micro_batch_size, config.seq_len + 1), generator=generator)
                    items = tokens[:, :-1].contiguous().to(device)
                    targets = tokens[:, 1:].contiguous().to(device)
                    groups = torch.randint(0, config.num_user_groups,
                                           (args.micro_batch_size,), generator=generator).to(device)
                    features = torch.randint(0, config.table_rows,
                                             (args.micro_batch_size, config.seq_len,
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
                metric = {"step": step, "host_step_ms": (time.perf_counter() - step_start) * 1000}
                # item() synchronizes only at log points; host_step_ms is not NPU kernel time.
                if step == 0 or (step + 1) % args.log_interval == 0:
                    metric["loss"] = loss.detach().float().cpu().item()
                    if rank == 0:
                        print(json.dumps(metric), flush=True)
                metrics.append(metric)
                if profiler is not None:
                    profiler.step()
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
        result = {"status": "completed", "rank": rank, "steps": args.steps,
                  "elapsed_s_synchronized": elapsed, "profile_enabled": args.profile,
                  "samples_per_s": args.steps * args.micro_batch_size * world_size / elapsed,
                  "profiled_elapsed_includes_profiler_overhead": args.profile}
        (rank_dir / "metrics.json").write_text(json.dumps({"summary": result, "steps": metrics}, indent=2))
        if args.save_checkpoint:
            engine.save_checkpoint(str(output_dir / "checkpoint"), tag="final")
        print(json.dumps(result), flush=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
