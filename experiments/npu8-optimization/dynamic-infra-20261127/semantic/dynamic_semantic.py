"""Audit eight dynamic eager training steps. This module is only loaded by semantic runs."""
import ctypes
import hashlib
import json
import math
from pathlib import Path

import torch


INJECTIONS = {2: ("nan", float("nan")), 4: ("posinf", float("inf")), 6: ("neginf", -float("inf"))}


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(tensor):
    value = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def checkpoint(optimizer):
    masters = optimizer.single_partition_of_fp32_groups
    states = [optimizer.optimizer.state.get(master, {}) for master in masters]
    return {
        "bf16_groups": [digest(group) for group in optimizer.bit16_groups_flat],
        "fp32_masters": [digest(master) for master in masters],
        "adamw": [{k: digest(v) if torch.is_tensor(v) else v for k, v in sorted(state.items())}
                  for state in states],
        "adamw_steps": [int(state.get("step", 0)) for state in states],
    }


def loaded_library():
    paths = {line.split(maxsplit=5)[-1].strip() for line in Path("/proc/self/maps").read_text().splitlines()
             if line.rstrip().endswith("/libtorch_npu.so")}
    check(len(paths) == 1, f"expected one loaded libtorch_npu.so: {paths}")
    path = Path(paths.pop())
    hash_state = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hash_state.update(chunk)
    library = ctypes.CDLL(str(path))
    counter = getattr(library, "npu_dynamic_index_rebase_counter", None)
    if counter is not None:
        counter.argtypes, counter.restype = [ctypes.c_uint], ctypes.c_uint64
    return {"loaded_library": str(path), "sha256": hash_state.hexdigest(),
            "counter": "npu_dynamic_index_rebase_counter" if counter is not None else None,
            "per_step": []}, counter


class SemanticAudit:
    def __init__(self, engine, output):
        self.engine = engine
        self.zero = engine.optimizer
        self.output = Path(output)
        self.records = []
        self.norms = []
        self.activation, self.counter = loaded_library()
        self.counter_names = ["eligible", "fallback", "key_uses", "descriptor_uses"]
        self.previous_counters = self.read_counters()
        self.activation["initial_counters"] = self.previous_counters
        original = self.zero.get_grad_norm_direct

        def audited_norm(*args, **kwargs):
            value = original(*args, **kwargs)
            self.norms.append(float(value))
            return value

        self.zero.get_grad_norm_direct = audited_norm

    def read_counters(self):
        if self.counter is None:
            return {}
        return {name: int(self.counter(i)) for i, name in enumerate(self.counter_names)}

    def step(self, index, items, targets, groups, features, loss):
        zero = self.zero
        case, sentinel = INJECTIONS.get(index, ("finite", None))
        invalid = sentinel is not None
        grads = [g for gid in range(len(zero.bit16_groups)) for g in zero.averaged_gradients[gid] if g is not None]
        check(len(grads) == 1052, f"step {index} saw {len(grads)} gradients")
        check(all(g.dtype == torch.bfloat16 and g.numel() for g in grads), "unexpected gradient partitions")
        loss_value = float(loss.detach().float().cpu())
        check(math.isfinite(loss_value), f"nonfinite loss before injection at step {index}")
        # Individual CPU reads keep the oracle independent of an NPU flatten/copy optimization.
        gradient_hashes = [digest(g) for g in grads]
        signature = [[list(g.shape), str(g.dtype)] for g in grads]
        before = checkpoint(zero)
        if self.records:
            check(self.records[-1]["after"] == before, f"optimizer state changed before step {index}")
        if invalid:
            target = grads[0] if case == "posinf" else grads[-1]
            with torch.no_grad():
                target.view(-1)[-1].fill_(sentinel)
        skipped_before = int(self.engine.skipped_steps)
        self.norms.clear()
        self.engine.step()
        torch.npu.synchronize()
        after = checkpoint(zero)
        skipped = int(self.engine.skipped_steps) - skipped_before
        applied = bool(self.engine.was_step_applied())
        check(bool(zero.overflow) == invalid and skipped == int(invalid) and applied == (not invalid),
              f"incorrect overflow decision on step {index}")
        if invalid:
            check(before == after and not self.norms, f"overflow step {index} updated state or entered norm")
        else:
            check(len(self.norms) == len(zero.bit16_groups) and all(math.isfinite(n) for n in self.norms),
                  f"missing finite gradient norm on step {index}")
            check(after["fp32_masters"] != before["fp32_masters"] and after["bf16_groups"] != before["bf16_groups"],
                  f"weights did not update on step {index}")
            check(after["adamw_steps"] == [n + 1 for n in before["adamw_steps"]] and
                  all({"exp_avg", "exp_avg_sq"} <= state.keys() for state in after["adamw"]),
                  f"AdamW did not update on step {index}")
        inputs = [items, targets, groups, features]
        record = {"index": index, "shape": list(items.shape), "case": case, "loss": loss_value,
                  "input_sha256": [digest(v) for v in inputs], "gradient_count": len(grads),
                  "gradient_layout_sha256": hashlib.sha256(json.dumps(signature).encode()).hexdigest(),
                  "gradient_sha256": gradient_hashes, "norms": list(self.norms), "overflow": bool(zero.overflow),
                  "skip_delta": skipped, "applied": applied, "before": before, "after": after}
        self.records.append(record)
        self.output.write_text(json.dumps(self.records, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        counters = self.read_counters()
        self.activation["per_step"].append({"step": index, "shape": list(items.shape), "cumulative": counters,
                                           "delta": {k: v - self.previous_counters[k] for k, v in counters.items()}})
        self.previous_counters = counters
        self.output.with_name("native_activation.json").write_text(json.dumps(self.activation, indent=2) + "\n",
                                                                  encoding="utf-8")
