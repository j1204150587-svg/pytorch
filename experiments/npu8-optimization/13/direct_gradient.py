"""Graph-owned BF16 gradients for the single-rank 11 + 09 training path."""

import torch
from torch.nn.modules import module as nn_module


def check_backward_hooks(model, parameters):
    if (nn_module._global_backward_hooks or nn_module._global_backward_pre_hooks
            or any(m._backward_hooks or m._backward_pre_hooks for m in model.modules())
            or any(p._backward_hooks or p._post_accumulate_grad_hooks for p in parameters)):
        raise RuntimeError("direct graph gradients require no user parameter or module backward hooks")


class DirectGradient:
    def __init__(self, zero, parameters, device, stream, offsets):
        self.zero = zero
        self.stream = stream
        self.device = device
        self.offsets = offsets  # Keep the pre-capture sidepack input alive.
        indices = {id(parameter): i for i, parameter in enumerate(parameters)}
        groups = zero.params_in_partition
        if (len(groups) != 1 or tuple(zero.groups_padding) != (0,) or len(parameters) != 1052
                or len(groups[0]) != len(indices) or set(map(id, groups[0])) != indices.keys()
                or zero.communication_data_type not in (torch.bfloat16, torch.float32)):
            raise RuntimeError("direct graph gradients require 1052 parameters in one unpadded BF16 partition")
        self.group = groups[0]
        self.order = tuple(indices[id(parameter)] for parameter in self.group)
        self.shapes = tuple(parameter.shape for parameter in self.group)
        self.sizes = tuple(parameter.numel() for parameter in self.group)
        self.numel = sum(self.sizes)
        self.generation = self.hits = 0
        self.phase = "idle"
        self.pending = None
        self.partition = None
        self.epilogue_installed = False

    def pack(self, gradients):
        pieces = []
        for index, shape in zip(self.order, self.shapes, strict=True):
            gradient = gradients[index]
            if (gradient is None or gradient.shape != shape or gradient.dtype != torch.bfloat16
                    or gradient.device != self.device or gradient.requires_grad or not gradient.is_contiguous()):
                raise RuntimeError(f"unsupported graph gradient layout at model parameter {index}")
            pieces.append(gradient.view(-1))
        return torch.cat(pieces)

    def bind(self, flat):
        if (self.partition is not None or flat.shape != (self.numel,) or flat.dtype != torch.bfloat16
                or flat.device != self.device or flat.requires_grad or not flat.is_contiguous()
                or (flat.device.type == "npu" and torch.ops.npu.get_npu_format(flat) != 2)):  # ACL_FORMAT_ND
            raise RuntimeError("expected a fresh, contiguous ND BF16 gradient buffer from capture")
        views = [part.view(shape) for part, shape in zip(flat.split(self.sizes), self.shapes, strict=True)]
        self.partition = views, flat

    def require_consumed(self):
        if self.pending is not None or self.phase not in ("idle", "consumed") or self.hits != self.generation:
            raise RuntimeError("direct graph gradients were not consumed exactly once")
        if self.generation and (self.zero._recsys_local_flat_gradients.get(0) is not self.partition
                                or self.zero.averaged_gradients.get(0) is not None
                                or self.zero.all_grad_tensors.get(0) is not None):
            raise RuntimeError("previous direct gradients are still live or were replaced before engine.step()")

    def begin_forward(self, stream):
        if (stream != self.stream or self.partition is None or self.zero._grad_acc_hooks
                or not self.epilogue_installed
                or self.zero.params_in_partition[0] is not self.group):
            raise RuntimeError("direct graph gradients require the original stream and the 09 partition epilogue")
        if not {"flatten_dense_tensors_aligned", "_update_model_bit16_weights",
                "get_grad_norm_direct", "has_overflow_partitioned_grads_serial"} <= vars(self.zero).keys():
            raise RuntimeError("direct graph gradients require the installed 09 and PTA optimizer paths")
        self.require_consumed()
        if self.generation:
            del self.zero._recsys_local_flat_gradients[0]
        self.generation += 1
        self.phase = "forward"
        return self.generation

    def begin_backward(self, generation, stream):
        if stream != self.stream or generation != self.generation or self.phase != "forward":
            raise RuntimeError("direct graph gradients require one matching backward on the forward stream")
        self.phase = "backward"

    def publish(self):
        if self.phase != "backward" or self.pending is not None:
            raise RuntimeError("direct graph gradients submitted twice or without backward")
        self.pending = self.partition
        self.phase = "ready"

    def consume(self, stream):
        if stream != self.stream or self.phase != "ready" or self.pending is None:
            raise RuntimeError("direct graph gradients missing, already consumed, or on another stream")
        partition = self.pending
        self.pending = None
        self.phase = "consumed"
        self.hits += 1
        return partition
