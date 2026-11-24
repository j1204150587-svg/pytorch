"""One BF16 partition, static scale 1, and one unfused FP32 AdamW group."""

from dataclasses import asdict, dataclass
import math

import torch


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


class FlatLayout:
    def __init__(self, model, partition):
        named = dict(model.named_parameters(remove_duplicate=False))
        names_by_id = {}
        for name, parameter in named.items():
            names_by_id.setdefault(id(parameter), []).append(name)
        ids = [id(parameter) for parameter in partition]
        require(len(ids) == len(set(ids)) and set(ids) == set(names_by_id),
                "partition must cover each distinct model parameter exactly once")
        require(all(p.requires_grad and p.numel() and p.is_contiguous()
                    and not getattr(p, "ds_pipe_replicated", False)
                    and not getattr(p, "model_parallel", False) for p in partition),
                "expected distinct contiguous trainable parameters without parallel replication")
        self.entries = []
        self.numel = 0
        for parameter in partition:
            self.entries.append((tuple(names_by_id[id(parameter)]), tuple(parameter.shape), self.numel,
                                 parameter.numel()))
            self.numel += parameter.numel()

    def views(self, flat):
        return [flat.narrow(0, offset, count).view(shape)
                for _, shape, offset, count in self.entries]

    def parameters(self, model):
        named = dict(model.named_parameters(remove_duplicate=False))
        require(named.keys() == {name for names, *_ in self.entries for name in names},
                "parameter names differ from the partition map")
        result = []
        for names, shape, *_ in self.entries:
            parameter = named[names[0]]
            require(tuple(parameter.shape) == shape and parameter.is_contiguous()
                    and all(named[name] is parameter for name in names),
                    f"changed shape, stride, or alias identity for {names}")
            result.append(parameter)
        require(len({id(p) for p in result}) == len(result), "unexpected new parameter aliases")
        return result

    def check_views(self, model, flat):
        for parameter, view in zip(self.parameters(model), self.views(flat), strict=True):
            require(parameter.data_ptr() == view.data_ptr() and parameter.device == flat.device
                    and parameter.dtype == flat.dtype and parameter.stride() == view.stride(),
                    "model parameter lost its flat BF16 view")

    def bind(self, model, flat):
        for parameter, view, (names, *_) in zip(
                self.parameters(model), self.views(flat), self.entries, strict=True):
            if parameter.device == flat.device:
                parameter.data = view
                continue
            replacement = torch.nn.Parameter(view, requires_grad=parameter.requires_grad)
            for name in names:
                parent, _, leaf = name.rpartition(".")
                setattr(model.get_submodule(parent), leaf, replacement)
        self.check_views(model, flat)


@dataclass
class Counters:
    global_steps: int = 0
    micro_steps: int = 0
    gas_boundary_ctr: int = 0
    global_samples: int = 0
    skipped_steps: int = 0
    successful_updates: int = 0


class SingleRankExecutor:
    def __init__(self, model, layout, bf16, master, adamw_options, clip_grad, batch_size, *,
                 captured_gradient=None, reuse_weight_views=False):
        require(bf16.shape == master.shape == (layout.numel,) and bf16.device == master.device
                and bf16.dtype == torch.bfloat16 and master.dtype == torch.float32
                and bf16.is_contiguous() and master.is_contiguous()
                and math.isfinite(clip_grad) and clip_grad > 0 and batch_size > 0,
                "expected one full contiguous BF16/FP32 partition and positive clipping/batch size")
        require(adamw_options.get("foreach") is False and not any(adamw_options.get(key) for key in
                ("fused", "amsgrad", "maximize", "capturable", "differentiable"))
                and all(isinstance(adamw_options[key], (int, float))
                        for key in ("lr", "eps", "weight_decay")),
                "expected reference unfused AdamW with scalar options")
        self.model, self.layout = model, layout
        self.bf16 = bf16.detach()
        self.master = torch.nn.Parameter(master.detach())
        self.optimizer = torch.optim.AdamW([self.master], **adamw_options)
        self.clip_grad, self.batch_size = clip_grad, batch_size
        self.counts = Counters()
        self.overflow, self.applied = False, False
        self.global_norm = 0.0
        self.pending, self.failed = False, False
        if captured_gradient is None:
            self.gradients = torch.empty_like(bf16, requires_grad=False)
            self._gradient_views = layout.views(self.gradients)
        else:
            self.bind_gradients(captured_gradient)
        if reuse_weight_views:
            layout.check_views(model, self.bf16)
        else:
            layout.bind(model, self.bf16)

    def bind_gradients(self, flat):
        """Adopt the fixed contiguous output of 13's backward capture before the first step."""
        require(not self.pending and not self.counts.global_steps and not self.failed,
                "gradient storage may only be bound before the first publication")
        require(flat.shape == self.bf16.shape and flat.device == self.bf16.device
                and flat.dtype == torch.bfloat16 and flat.is_contiguous() and not flat.requires_grad
                and flat.untyped_storage().data_ptr() != self.bf16.untyped_storage().data_ptr(),
                "expected separate fixed BF16 gradient storage")
        self.gradients = flat
        self._gradient_views = self.layout.views(flat)

    def publish(self):
        require(not self.pending and not self.failed, "gradient publication is duplicate or executor failed")
        self.pending = True

    @torch.no_grad()
    def step(self, statistics=None):
        require(self.pending and not self.failed, "step requires one fresh gradient publication")
        self.pending = False
        try:
            if statistics is None:
                floating = self.gradients.float()
                self.overflow = bool(floating.isnan().logical_or(floating.isinf()).any().item())
            else:
                self.overflow = statistics.has_overflow()
            self.applied = not self.overflow
            clipped = None
            if self.applied:
                if statistics is None:
                    # ZeRO-2 uses one FP64 norm per shard and casts the square sum to FP32.
                    norms = [torch.linalg.vector_norm(shard.double(), ord=2) for shard in self._gradient_views]
                    group_norm = torch.stack(norms).square().sum().float().pow(0.5)
                    group_norm.masked_fill_(group_norm.isnan().logical_or(group_norm.isinf()), -1.0)
                    scaled_norm = torch.linalg.vector_norm(group_norm.reshape(1), ord=2)
                else:
                    scaled_norm = statistics.scaled_norm()
                self.global_norm = scaled_norm / 1.0
                if statistics is None:
                    clip = torch.clamp(((scaled_norm / 1.0) + 1e-6) / self.clip_grad, min=1.0)
                    clipped = floating.mul_(1.0 / (clip * 1.0))
                else:
                    clipped = self.gradients.to(self.master.dtype)
                    statistics.clip(clipped, scaled_norm)
                self.master.grad = clipped
                self.optimizer.step()
                self.bf16.copy_(self.master)
            self.counts.global_steps += 1
            self.counts.micro_steps += 1
            self.counts.gas_boundary_ctr += 1
            self.counts.global_samples += self.batch_size
            self.counts.skipped_steps += int(self.overflow)
            self.counts.successful_updates += int(self.applied)
            return clipped
        except Exception:
            self.failed = True
            raise
        finally:
            self.master.grad = None

    def state_dict(self):
        require(not self.pending and not self.failed, "checkpoint needs a completed step boundary")
        self.layout.check_views(self.model, self.bf16)
        return {"layout": self.layout.entries, "master": self.master.detach(), "bf16": self.bf16,
                "adamw": self.optimizer.state_dict(), "counts": asdict(self.counts),
                "decision": (self.overflow, self.applied), "norm": float(self.global_norm),
                "clip_grad": self.clip_grad, "batch_size": self.batch_size}

    @torch.no_grad()
    def load_state_dict(self, state):
        require(not self.pending and not self.failed and not self.counts.global_steps,
                "restore requires a fresh executor")
        require(state["layout"] == self.layout.entries and state["clip_grad"] == self.clip_grad
                and state["batch_size"] == self.batch_size, "checkpoint partition or config changed")
        for key, target in (("master", self.master), ("bf16", self.bf16)):
            value = state[key]
            require(value.shape == target.shape and value.dtype == target.dtype,
                    f"checkpoint changed the layout of {key}")
        counts = Counters(**state["counts"])
        opt = state["adamw"]
        groups = opt["param_groups"]
        require(len(groups) == 1 and len(groups[0]["params"]) == 1
                and {key: value for key, value in groups[0].items() if key != "params"} ==
                {key: value for key, value in self.optimizer.param_groups[0].items() if key != "params"},
                "checkpoint AdamW options changed")
        moments = opt["state"].get(groups[0]["params"][0], {})
        require(int(moments.get("step", 0)) == counts.successful_updates
                and counts.successful_updates + counts.skipped_steps == counts.global_steps
                and counts.global_steps == counts.micro_steps == counts.gas_boundary_ctr
                and counts.global_samples == counts.global_steps * self.batch_size,
                "checkpoint counters disagree")
        if moments:
            require(moments.keys() == {"step", "exp_avg", "exp_avg_sq"}
                    and all(moments[key].shape == self.master.shape and moments[key].dtype == torch.float32
                            for key in ("exp_avg", "exp_avg_sq")), "invalid checkpoint moments")
        self.master.copy_(state["master"])
        self.bf16.copy_(state["bf16"])
        self.optimizer.load_state_dict(opt)
        self.counts = counts
        self.overflow, self.applied = state["decision"]
        self.global_norm = state["norm"]
        self.layout.check_views(self.model, self.bf16)
