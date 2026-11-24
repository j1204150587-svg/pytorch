"""Transfer the 26 state and consume its backward graph output at the update boundary."""

from dataclasses import asdict
import hashlib
import inspect
import json
import textwrap

import torch

from direct_gradient import DirectGradient
from semantic_gate import STEP_SHA256
from single_rank import FlatLayout, SingleRankExecutor, require


class ExecutorGradient(DirectGradient):
    """Use the original capture, pack, and backward publish protocol."""

    def require_consumed(self):
        require(self.pending is None and self.phase in ("idle", "consumed")
                and self.hits == self.generation == self.executor.counts.global_steps
                and not self.executor.pending and not self.executor.failed,
                "previous replay generation was not consumed once by the executor")

    def begin_forward(self, stream):
        require(stream == self.stream and self.partition is not None and not self.zero._grad_acc_hooks
                and self.epilogue_installed and self.zero.params_in_partition[0] is self.group,
                "executor replay requires the original training stream and partition layout")
        self.require_consumed()
        self.generation += 1
        self.phase = "forward"
        return self.generation


class Statistics26:
    def __init__(self, zero, direct):
        self.zero, self.direct = zero, direct

    def has_overflow(self):
        # Reuse ZeRO's byte conversion, rank-one MAX and Host scalar read.
        self.zero.check_overflow(partition_gradients=True)
        self.zero._update_scale(self.zero.overflow)
        return self.zero.overflow

    def scaled_norm(self):
        group_norm = self.zero.get_grad_norm_direct(self.direct.partition[0], self.direct.group)
        return torch.linalg.vector_norm(torch.stack([group_norm]), ord=2)

    def clip(self, gradient, scaled_norm):
        self.zero.unscale_and_clip_grads([gradient], scaled_norm)


class Owner26:
    def __init__(self, engine, direct, current_stream):
        self.engine, self.direct, self.current_stream = engine, direct, current_stream
        zero = self.zero = engine.optimizer
        group, = zero.optimizer.param_groups
        master, = zero.single_partition_of_fp32_groups
        bf16, = zero.bit16_groups_flat
        (partition,), = zero.parallel_partitioned_bit16_groups
        require(engine.basic_optimizer is zero.optimizer and type(zero.optimizer) is torch.optim.AdamW
                and len(group["params"]) == 1 and group["params"][0] is master
                and master.grad is None and not zero.optimizer.state and not engine.global_steps
                and bf16.data_ptr() == partition.data_ptr() and bf16.shape == partition.shape,
                "state transfer requires the fresh 26 weight and AdamW partition")
        layout = FlatLayout(engine.module, direct.group)
        require(any(names == ("item_embedding.weight", "output.weight") for names, *_ in layout.entries),
                "missing HSTU tied output weight")
        options = inspect.signature(torch.optim.AdamW).parameters.keys()
        options = {key: group[key] for key in options if key in group and key != "params"}
        self.executor = SingleRankExecutor(
            engine.module, layout, bf16, master, options, zero.clip_grad, engine.train_batch_size(),
            captured_gradient=direct.partition[1], reuse_weight_views=True,
        )
        require(self.executor.master.data_ptr() == master.data_ptr(), "master moved during transfer")
        direct.executor = self.executor
        self.statistics = Statistics26(zero, direct)
        self.epilogue_hits = 0

    def activate(self):
        zero = self.zero
        # Release ZeRO's ownership while retaining the public Engine runtime shell.
        for parameter in self.direct.group:
            parameter._hp_mapping = None
        zero.optimizer.param_groups[0]["params"] = []
        zero.optimizer.step = self._old_optimizer_step
        zero.single_partition_of_fp32_groups.clear()
        zero.bit16_groups_flat.clear()
        zero.parallel_partitioned_bit16_groups.clear()
        del zero._recsys_local_flat_gradients
        zero.overlapping_partition_gradients_reduce_epilogue = self.epilogue
        zero.step = self.step

    @staticmethod
    def _old_optimizer_step(*_args, **_kwargs):
        raise RuntimeError("the old ZeRO AdamW no longer owns any parameters")

    def epilogue(self):
        zero = self.zero
        require(zero.is_gradient_accumulation_boundary and not zero._epilogue_ran_this_backward
                and self.direct.phase == "ready" and self.epilogue_hits == self.direct.hits,
                "expected one backward publication at the 26 accumulation boundary")
        zero._epilogue_ran_this_backward = True
        self.epilogue_hits += 1

    def step(self, closure=None):
        require(closure is None and self.epilogue_hits == self.direct.generation
                and self.epilogue_hits == self.direct.hits + 1
                and self.zero._epilogue_ran_this_backward, "expected one completed backward before step")
        _, flat = self.direct.consume(self.current_stream(self.direct.device))
        require(flat is self.executor.gradients, "replay gradient does not match the executor capture")
        self.zero.micro_step_id = -1
        self.executor.publish()
        self.executor.step(self.statistics)
        self.zero._global_grad_norm = self.executor.global_norm

    def finish(self, steps, rank_dir):
        self.direct.require_consumed()
        self.executor.layout.check_views(self.engine.module, self.executor.bf16)
        counts = asdict(self.executor.counts)
        zero = self.zero
        require(self.epilogue_hits == self.direct.hits == steps == counts["global_steps"]
                and all(getattr(self.engine, key) == value
                        for key, value in counts.items() if key != "successful_updates")
                and counts["successful_updates"] == steps - counts["skipped_steps"]
                and not zero.single_partition_of_fp32_groups and not zero.bit16_groups_flat
                and not zero.parallel_partitioned_bit16_groups
                and not zero.optimizer.param_groups[0]["params"] and not zero.optimizer.state
                and all(value is None for value in zero.averaged_gradients.values())
                and all(value is None for value in zero.all_grad_tensors.values()),
                "executor state, counters or ZeRO gradient slots disagree")
        (rank_dir / "state_owner.json").write_text(json.dumps({
            "owner": "single_rank_executor", "epilogue_hits": self.epilogue_hits,
            "consumed": self.direct.hits, "counts": counts,
            "flat_numel": self.executor.layout.numel, "parameters": len(self.direct.group),
        }, indent=2) + "\n", encoding="utf-8")


def attach(engine, device, rank_dir, norm_stats, install_pta):
    import deepspeed
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    zero = engine.optimizer
    direct = zero._recsys_direct_gradient
    source = textwrap.dedent(inspect.getsource(zero.step)).strip()
    require(type(zero) is DeepSpeedZeroOptimizer and type(direct) is ExecutorGradient
            and deepspeed.__version__ == "0.19.0" and "step" not in vars(zero)
            and hashlib.sha256(source.encode()).hexdigest() == STEP_SHA256
            and engine.lr_scheduler is None and engine.quantizer is None
            and zero.loss_scale == 1.0 and not zero.dynamic_loss_scale and not zero.custom_loss_scaler
            and direct.epilogue_installed and not zero._grad_acc_hooks
            and torch.device(zero.device) == device, "unsupported 26 optimizer runtime")
    owner = Owner26(engine, direct, torch.npu.current_stream)
    finish_pta = install_pta(engine, rank_dir, norm_stats, direct_owner=direct)
    owner.activate()
    return owner, finish_pta
