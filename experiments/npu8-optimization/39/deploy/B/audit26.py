"""Opt-in untimed, per-step correctness audit of the 26 A/B training entries."""

import math

import torch

from compare26 import CASES
from semantic_gate import digest, update_state, write_json
from single_rank import FlatLayout, require


class Audit26:
    def __init__(self, engine, owner, rank_dir, steps):
        require(steps == len(CASES), "unexpected correctness case count")
        self.engine, self.owner = engine, owner
        self.zero = engine.optimizer
        self.direct = self.zero._recsys_direct_gradient
        if owner is None:
            self.master, = self.zero.single_partition_of_fp32_groups
            self.bf16, = self.zero.bit16_groups_flat
            self.optimizer = self.zero.optimizer
            self.layout = FlatLayout(engine.module, self.direct.group)
        else:
            executor = owner.executor
            self.master, self.bf16, self.optimizer, self.layout = (
                executor.master, executor.bf16, executor.optimizer, executor.layout)
        self.path = rank_dir / "integration_correctness.json"
        self.report = {"status": "running", "owner": "B" if owner else "A",
                       "layout": self.layout.entries, "steps": []}
        self.row = None
        original_clip = self.zero.unscale_and_clip_grads

        def record_clip(groups, norm):
            original_clip(groups, norm)
            require(self.row is not None and "clipped_fp32" not in self.row and len(groups) == 1,
                    "expected one FP32 gradient clipping call")
            scaled_norm = float(norm)
            require(math.isfinite(scaled_norm), "non-finite clip norm")
            self.row.update(clipped_fp32=digest(groups[0]), scaled_norm=scaled_norm)

        self.zero.unscale_and_clip_grads = record_clip
        write_json(self.path, self.report)

    @torch.no_grad()
    def before(self, step, loss):
        require(self.row is None and step == len(self.report["steps"]), "audit step order changed")
        require(self.direct.phase == ("ready" if self.owner else "consumed"),
                "the backward graph did not publish its gradient before the audit")
        case = CASES[step]
        self.layout.check_views(self.engine.module, self.bf16)
        views, gradient = self.direct.partition
        self.row = {"case": case, "before": update_state(self.master, self.bf16, self.optimizer),
                    "loss": float(loss), "raw_bf16": digest(gradient)}
        require(math.isfinite(self.row["loss"]), "non-finite HSTU forward loss")
        if step < 6:
            gradient.fill_(2 ** -14)
            injected = {"initial_nan": math.nan, "nan": math.nan, "posinf": math.inf,
                        "neginf": -math.inf, "finite_recovery_above_clip": 16.0}
            if case in injected:
                views[0 if case == "posinf" else -1].view(-1)[-1].fill_(injected[case])
        self.row["optimizer_input_bf16"] = digest(gradient)
        self.report["steps"].append(self.row)
        write_json(self.path, self.report)

    def after(self):
        row = self.row
        self.layout.check_views(self.engine.module, self.bf16)
        after = update_state(self.master, self.bf16, self.optimizer)
        overflow = row["case"] in ("initial_nan", "nan", "posinf", "neginf")
        counters = {key: getattr(self.engine, key) for key in
                    ("global_steps", "micro_steps", "global_samples", "gas_boundary_ctr", "skipped_steps")}
        step = len(self.report["steps"])
        skips = sum(k in ("initial_nan", "nan", "posinf", "neginf") for k in CASES[:step])
        require(counters == {"global_steps": step, "micro_steps": step, "gas_boundary_ctr": step,
                             "global_samples": step * self.engine.train_batch_size(), "skipped_steps": skips}
                and after["adam_step"] == step - skips and overflow == self.zero.overflow
                and self.engine.was_step_applied() == (not overflow), "overflow, apply or step counters differ")
        require(all(p.grad is None for p in self.engine.module.parameters())
                and all(g is None for g in self.zero.averaged_gradients.values())
                and all(g is None for g in self.zero.all_grad_tensors.values()), "live gradient after step")
        self.direct.require_consumed()
        require(digest(self.master.detach().to(torch.bfloat16)) == after["bf16"], "bad BF16 writeback")
        if overflow:
            require(after == row["before"] and "clipped_fp32" not in row, "overflow changed optimizer state")
        else:
            require("clipped_fp32" in row and after["master"] != row["before"]["master"],
                    "finite step did not update master")
            if row["case"] in ("finite_below_clip", "finite_recovery_above_clip"):
                above = (row["scaled_norm"] + 1e-6) > self.zero.clip_grad
                require(above == (row["case"] == "finite_recovery_above_clip"), "wrong side of clip threshold")
        row.update(after=after, counters=counters, overflow=overflow, applied=not overflow)
        self.row = None
        write_json(self.path, self.report)

    def finish(self):
        require(self.row is None and tuple(row["case"] for row in self.report["steps"]) == CASES,
                "integration audit incomplete")
        self.report["status"] = "passed"
        write_json(self.path, self.report)
