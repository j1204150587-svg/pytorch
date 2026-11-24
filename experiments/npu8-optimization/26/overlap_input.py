"""Single-threaded lookahead for the original 13 graph input staging path."""

import ctypes
from dataclasses import dataclass
import hashlib
from threading import get_ident

import torch


# Keep the buffers alive if a device error prevents a completion guarantee.
_undrained = []


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for tensor in tensors:
        if tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise ValueError("input audit requires contiguous CPU tensors")
        digest.update(repr((tuple(tensor.shape), tensor.dtype)).encode("ascii"))
        digest.update(ctypes.string_at(tensor.data_ptr(), tensor.numel() * tensor.element_size()))
    return digest.hexdigest()


def fill_cpu_inputs(generator, config, tokens, inputs):
    items, targets, groups, features = inputs
    torch.randint(1, config.vocab_size, tokens.shape, generator=generator, out=tokens)
    items.copy_(tokens[:, :-1])
    targets.copy_(tokens[:, 1:])
    torch.randint(0, config.num_user_groups, groups.shape, generator=generator, out=groups)
    torch.randint(0, config.table_rows, features.shape, generator=generator, out=features)


@dataclass
class _Slot:
    host: tuple
    device: tuple
    ready: object = None
    used: object = None


class OverlapInput:
    def __init__(self, npu, device, config, batch_size, generator):
        self.npu = npu
        self.device = device
        self.training = npu.current_stream(device)
        self.transfer = npu.Stream(device=device)
        self.thread = get_ident()
        self.config = config
        self.generator = generator
        shape = (batch_size, config.seq_len)
        shapes = (shape, shape, (batch_size,), (*shape, config.num_tables, config.features_per_table))
        self.tokens = torch.empty((batch_size, config.seq_len + 1), dtype=torch.long, device="cpu")
        self.slots = []
        with npu.stream(self.transfer):
            for _ in range(2):
                host = tuple(torch.empty(s, dtype=torch.long, device="cpu").pin_memory() for s in shapes)
                if not all(t.is_pinned() for t in host):
                    raise RuntimeError("overlap input requires NPU-pinned host buffers")
                dst = tuple(torch.empty(s, dtype=torch.long, device=device) for s in shapes)
                self.slots.append(_Slot(host, dst))
        self.pending = None
        self.active = None
        self.next_slot = 0
        self.submitted = 0
        self.consumed = 0

    def _check_context(self):
        if get_ident() != self.thread or self.npu.current_stream(self.device) != self.training:
            raise RuntimeError("overlap input requires the original training thread and stream")

    def take(self, step):
        self._check_context()
        if self.active is not None:
            raise RuntimeError("previous overlap input is still in use")
        if self.pending is None:
            return None
        expected, slot = self.pending
        if expected != step:
            raise RuntimeError("prefetched input does not match the training step")
        self.training.wait_event(slot.ready)
        self.pending = None
        self.active = slot
        self.consumed += 1
        return slot.device

    def release(self):
        self._check_context()
        if self.active is not None:
            # The training stream has queued all original input uses, including
            # the eager loss backward's use of targets.
            self.active.used = self.training.record_event()
            self.active = None

    def submit(self, step, audit=None):
        self._check_context()
        if self.active is not None or self.pending is not None:
            raise RuntimeError("overlap input already active or pending")
        slot = self.slots[self.next_slot]
        self.next_slot ^= 1
        if slot.ready is not None and not slot.ready.query():
            slot.ready.synchronize()
        fill_cpu_inputs(self.generator, self.config, self.tokens, slot.host)
        if audit is not None:
            audit.append(tensor_digest(slot.host))
        with self.npu.stream(self.transfer):
            if slot.used is not None:
                self.transfer.wait_event(slot.used)
            for dst, host in zip(slot.device, slot.host, strict=True):
                dst.copy_(host, non_blocking=True)
            slot.ready = self.transfer.record_event()
        self.pending = (step, slot)
        self.submitted += 1

    def verify_finished(self, expected):
        if (self.pending is not None or self.active is not None
                or self.submitted != expected or self.consumed != expected):
            raise RuntimeError("overlap input usage disagrees with the training loop")

    def drain_after_failure(self):
        try:
            self.npu.synchronize(self.device)
        except BaseException:
            _undrained.append(self)
            raise
