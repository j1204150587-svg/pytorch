"""Compare the candidate against each step of the stock or 26 ZeRO-2 reference."""

import ctypes
from dataclasses import asdict
import hashlib
import inspect
import io
import json
import math
from pathlib import Path
import textwrap

import torch

from single_rank import FlatLayout, SingleRankExecutor, require


STEP_SHA256 = "44750b4ab49af6bb844cb020d3c2306f474a32c90ff4e7d56b140b75187dc966"


def write_json(path, value):
    if path is not None:
        path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(tensor):
    cpu = tensor.detach().cpu().contiguous()
    raw = ctypes.string_at(cpu.data_ptr(), cpu.numel() * cpu.element_size())
    return [str(tensor.dtype), list(tensor.shape), hashlib.sha256(raw).hexdigest()]


def update_state(master, bf16, optimizer):
    state = optimizer.state.get(master, {})
    require(master.dtype == torch.float32 and bf16.dtype == torch.bfloat16 and master.grad is None,
            "expected FP32 master, BF16 weights and an empty master .grad")
    require(not state or state.keys() == {"step", "exp_avg", "exp_avg_sq"},
            "unexpected AdamW state fields")
    if state:
        require(all(state[key].dtype == torch.float32 and state[key].shape == master.shape
                    and state[key].device == master.device for key in ("exp_avg", "exp_avg_sq")),
                "expected full FP32 AdamW moments on the master device")
    return {"master": digest(master), "bf16": digest(bf16),
            "adamw": {key: digest(value) for key, value in state.items()},
            "adam_step": int(state.get("step", 0))}


class SemanticGate:
    def __init__(self, engine, model_factory, reference, output=None, *, live=True):
        self.engine, self.model_factory = engine, model_factory
        zero = self.zero = engine.optimizer
        if live:
            import deepspeed
            from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

            source = textwrap.dedent(inspect.getsource(zero.step)).strip()
            require(deepspeed.__version__ == "0.19.0" and type(zero) is DeepSpeedZeroOptimizer
                    and "step" not in vars(zero) and hashlib.sha256(source.encode()).hexdigest() == STEP_SHA256,
                    "expected the stock DeepSpeed 0.19.0 ZeRO-2 step used by 13/26")
            require(engine.world_size == 1 and engine.gradient_accumulation_steps() == 1
                    and zero.partition_gradients and zero.check_grad_overflow
                    and not zero.dynamic_loss_scale and not zero.custom_loss_scaler and zero.loss_scale == 1.0
                    and zero.dtype == zero.gradient_accumulation_dtype == torch.bfloat16
                    and zero.model_parallel_group is None and not zero.has_moe_layers and not zero.cpu_offload
                    and not zero.round_robin_gradients and not zero.zenflow and zero.torch_autocast_gradscaler is None
                    and engine.lr_scheduler is None, "unsupported ZeRO-2 configuration")
        partition, = zero.params_in_partition
        self.master, = zero.single_partition_of_fp32_groups
        self.bf16, = zero.bit16_groups_flat
        (low_partition,), = zero.parallel_partitioned_bit16_groups
        self.adamw = zero.optimizer
        (group,) = self.adamw.param_groups
        require(tuple(zero.groups_padding) == (0,) and type(self.adamw) is torch.optim.AdamW
                and len(group["params"]) == 1 and group["params"][0] is self.master
                and low_partition.data_ptr() == self.bf16.data_ptr() and low_partition.shape == self.bf16.shape
                and self.master.grad is None and not self.adamw.state, "expected a fresh unpadded ZeRO-2 partition")
        self.layout = FlatLayout(engine.module, partition)
        aliases = [list(names) for names, *_ in self.layout.entries if len(names) > 1]
        require(["item_embedding.weight", "output.weight"] in aliases, "HSTU tied output/embedding alias is missing")
        self.layout.check_views(engine.module, self.bf16)
        options = inspect.signature(torch.optim.AdamW).parameters.keys()
        self.adamw_options = {key: value for key, value in group.items() if key != "params" and key in options}
        self.executor = self.new_executor()
        self.output = Path(output) if output is not None else None
        self.report = {"status": "running", "reference": reference, "torch": str(torch.__version__),
                       "layout": {"numel": self.layout.numel, "parameters": len(partition), "aliases": aliases},
                       "steps": []}
        self.row = None
        write_json(self.output, self.report)

    def new_executor(self):
        # Module construction is CPU-only. Weight storage is replaced before its first use.
        with torch.random.fork_rng(devices=[]):
            model = self.model_factory()
        return SingleRankExecutor(model, self.layout, self.bf16.clone(), self.master.detach().clone(),
                                  self.adamw_options, self.zero.clip_grad, self.engine.train_batch_size())

    def checkpoint_roundtrip(self):
        former = self.executor
        checkpoint = io.BytesIO()
        torch.save(former.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = self.new_executor()
        with torch.no_grad():
            restored.master.zero_()
            restored.bf16.zero_()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        require(update_state(former.master, former.bf16, former.optimizer) ==
                update_state(restored.master, restored.bf16, restored.optimizer),
                "checkpoint restore differs from its source")
        self.executor = restored
        self.report["checkpoint_roundtrip"] = "passed"

    def begin(self, case, gradients, *, expected_overflow, expected_clip=None):
        require(self.row is None, "previous gate step was not compared")
        executor = self.executor
        self.layout.check_views(self.engine.module, self.bf16)
        self.layout.check_views(executor.model, executor.bf16)
        before = update_state(self.master, self.bf16, self.adamw)
        require(before == update_state(executor.master, executor.bf16, executor.optimizer),
                f"{case}: reference and candidate disagree before the step")
        targets = executor.layout.views(executor.gradients)
        require(len(targets) == len(gradients), "gradient shard count changed")
        with torch.no_grad():
            for target, grad in zip(targets, gradients, strict=True):
                require(grad.shape == target.shape and grad.dtype == target.dtype
                        and grad.device == target.device and grad.is_contiguous(),
                        "gradient layout differs from the live parameter map")
                target.copy_(grad)
        gradients_digest = digest(executor.gradients)
        self.row = {"case": case, "before": before, "gradient": gradients_digest,
                    "expected_overflow": expected_overflow, "expected_clip": expected_clip}
        self.report["steps"].append(self.row)
        write_json(self.output, self.report)
        executor.publish()
        clipped = executor.step()
        require(digest(executor.gradients) == gradients_digest, f"{case}: candidate modified the BF16 gradient")
        self.row["candidate_clipped"] = digest(clipped) if clipped is not None else None

    def record_clip(self, flat, norm):
        require(self.row is not None and "reference_clipped" not in self.row,
                "reference clipped outside the expected step")
        self.row.update(reference_clipped=digest(flat), reference_norm=float(norm))

    def finish_step(self):
        row, executor, engine = self.row, self.executor, self.engine
        case = row["case"]
        ref = update_state(self.master, self.bf16, self.adamw)
        actual = update_state(executor.master, executor.bf16, executor.optimizer)
        counters = {key: getattr(engine, key) for key in
                    ("global_steps", "micro_steps", "gas_boundary_ctr", "global_samples", "skipped_steps")}
        counters["successful_updates"] = counters["global_steps"] - counters["skipped_steps"]
        row.update(reference_after=ref, candidate_after=actual, reference_counts=counters,
                   candidate_counts=asdict(executor.counts), overflow=bool(self.zero.overflow),
                   applied=bool(engine.was_step_applied()), candidate_overflow=executor.overflow,
                   candidate_applied=executor.applied,
                   candidate_norm=float(executor.global_norm), reference_global_norm=float(self.zero._global_grad_norm))
        write_json(self.output, self.report)
        self.layout.check_views(engine.module, self.bf16)
        self.layout.check_views(executor.model, executor.bf16)
        require(all(parameter.grad is None for parameter in engine.module.parameters())
                and all(parameter.grad is None for parameter in executor.model.parameters())
                and all(value is None for value in self.zero.averaged_gradients.values())
                and all(value is None for value in self.zero.all_grad_tensors.values())
                and not executor.pending, f"{case}: gradient ownership did not return to idle")
        require(row["overflow"] == executor.overflow == row["expected_overflow"]
                and row["applied"] == executor.applied == (not executor.overflow),
                f"{case}: overflow/apply decision differs")
        require(counters == row["candidate_counts"] and actual["adam_step"] == counters["successful_updates"],
                f"{case}: engine counters or AdamW step differ")
        require(row["candidate_clipped"] == row.get("reference_clipped"),
                f"{case}: clipped FP32 gradient differs")
        require(math.isclose(row["candidate_norm"], row["reference_global_norm"], rel_tol=1e-5, abs_tol=1e-7),
                f"{case}: global norm differs")
        require(actual == ref,
                f"{case}: state differs at " + ", ".join(key for key in actual if actual[key] != ref[key]))
        if executor.overflow:
            require(actual == row["before"] and "reference_clipped" not in row,
                    f"{case}: overflow changed state or clipped gradients")
        else:
            require(actual["master"] != row["before"]["master"] and actual["adamw"],
                    f"{case}: finite gradients did not update master/AdamW")
            if row["expected_clip"] is not None:
                require(((row["reference_norm"] + 1e-6) > executor.clip_grad) == row["expected_clip"],
                        f"{case}: test gradient on the wrong side of the clipping threshold")
        require(digest(executor.master.detach().to(torch.bfloat16)) == actual["bf16"],
                f"{case}: BF16 writeback differs from the FP32 master cast")
        row["status"] = "passed"
        self.row = None

    def finish(self, expected_cases):
        require(self.row is None and [row["case"] for row in self.report["steps"]] == list(expected_cases)
                and all(row["status"] == "passed" for row in self.report["steps"]),
                "the gate did not compare every requested step")
        self.report["status"] = "passed"
        write_json(self.output, self.report)

    def fail(self, error):
        self.report.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(self.output, self.report)
