"""Use the PR's NPU device guard in generated-wrapper controls on the stock 2.10 backend."""
def install():
    import torch_npu
    import torch_npu._inductor
    from torch._inductor.codegen.common import get_device_op_overrides, register_device_op_overrides
    from torch_npu._inductor.codegen.npu.device_op_overrides import NewNPUDeviceOpOverrides
    before = get_device_op_overrides("npu")
    original = before.device_guard(0)
    upstream = NewNPUDeviceOpOverrides()
    register_device_op_overrides("npu", upstream)
    expected = upstream.device_guard(0)
    if expected != "torch.npu.utils.device(0)":
        raise RuntimeError("unexpected PR device guard")
    # The host torch 2.10 wrapper can also emit the private generic spelling
    # independently of the registered override. Use the exact PTA guard class.
    compat = not hasattr(torch_npu.npu, "_DeviceGuard")
    if compat:
        torch_npu.npu._DeviceGuard = torch_npu.npu.utils.device
    return {"original": original, "registered": expected, "generic_guard_alias": compat}
