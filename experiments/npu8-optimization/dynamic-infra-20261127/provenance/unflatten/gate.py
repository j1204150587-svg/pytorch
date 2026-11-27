"""Check the CPU view construction, or run the opt-in compiled NPU operator gate."""
import argparse
import json
import os
import torch

native = torch._C._nn.unflatten_dense_tensors
MARK = "NPUUnflattenContiguousViews"


def native_reference(flat, refs):
    # PyTorch v2.10.0 aten/src/ATen/native/TensorShape.cpp:4800
    outputs = []
    offset = 0
    for shape in refs:
        n = shape.numel()
        if n:
            outputs.append(flat.narrow(0, offset, n).view(shape.shape))
            offset += n
        else:
            outputs.append(torch.empty(shape.shape, dtype=flat.dtype, device=flat.device))
    return outputs


def candidate_cpu(flat, refs):
    offset = flat.storage_offset()
    outputs = []
    for shape in refs:
        dims = shape.shape
        step = 1
        strides = [0] * len(dims)
        for i in reversed(range(len(dims))):
            strides[i] = step
            step *= max(1, dims[i])
        outputs.append(flat.as_strided(dims, strides, offset))
        offset += shape.numel()
    return outputs


def checked_call(flat, refs, expect_fast, device):
    if device == "cpu":
        return candidate_cpu(flat, refs) if expect_fast else native(flat, refs)
    # CPU profiler scope records the C++ marker without adding a device sync.
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        outputs = native(flat, refs)
    events = prof.events()
    hits = sum(e.name == MARK for e in events)
    active = os.environ.get("NPU_EXPERIMENT_UNFLATTEN_VIEWS") == "1"
    assert hits == int(expect_fast and active), (expect_fast, hits, [e.name for e in events[:12]])
    assert any(e.name == "aten::unflatten_dense_tensors" for e in events)
    return outputs


def gate(device):
    if device == "npu":
        import torch_npu
        active = os.environ.get("NPU_EXPERIMENT_UNFLATTEN_VIEWS") == "1"
        actual = torch._C._dispatch_has_kernel_for_dispatch_key(
            "aten::unflatten_dense_tensors", "AutogradPrivateUse1")
        assert actual == active, (actual, active)
    dtype = torch.bfloat16
    refs = [torch.empty(s, device=device, dtype=dtype) for s in [(2,3), (), (4,1), (3,2,1)] * 9]
    # The upstream operator reads only reference sizes. Include noncontiguous
    # templates and parameters requiring grad as in group metadata.
    refs[0] = torch.empty((3,2), device=device, dtype=dtype).t()
    refs[1] = torch.nn.Parameter(refs[1])
    n = sum(x.numel() for x in refs)
    def footprint(t):
        return t.shape, t.stride(), t.storage_offset(), t.dtype, t.device, t.is_leaf, t.requires_grad
    def makeflat(extra=0):
        return (torch.arange(n + extra, dtype=torch.float32, device=device) % 67).to(dtype)
    results = {}
    for gradmode in (True, False):
        with torch.set_grad_enabled(gradmode):
            a = makeflat()
            b = a.clone()
            expected = native_reference(a, refs)
            result = checked_call(b, refs, True, device)
            again = checked_call(b, refs, True, device)
            for i, (left, right) in enumerate(zip(expected,result,strict=True)):
                assert footprint(left) == footprint(right)
                assert left._base is a and right._base is b and right._cdata != again[i]._cdata
                assert left.untyped_storage()._cdata == a.untyped_storage()._cdata
                assert right.untyped_storage()._cdata == b.untyped_storage()._cdata
                assert torch.equal(left, right)
            a_version, b_version = a._version, b._version
            expected[0].fill_(11)
            result[0].fill_(11)
            assert torch.equal(a,b) and torch.equal(again[0],expected[0])
            assert a._version - a_version == b._version - b_version == 1
            assert all(x._version == b._version for x in result)
            expected[0].resize_(3,2)
            result[0].resize_(3,2)
            assert footprint(expected[0]) == footprint(result[0])
            assert again[0].shape == refs[0].shape
            assert a._version - a_version == b._version - b_version
            # A second flat must produce independent aliases at its own address.
            c = makeflat()
            fresh = checked_call(c, refs, True, device)
            assert fresh[0]._base is c and fresh[0].data_ptr() != result[0].data_ptr()
            results[f"finite_grad_mode_{gradmode}"] = len(result)
    root = makeflat(5)
    for label,flat,shapes in (("offset",root[1:1+n],refs),
                              ("trailing_padding",root,refs),
                              ("prefix",makeflat(),refs[:-1])):
        actual = checked_call(flat,shapes,True,device)
        expected = native_reference(flat,shapes)
        assert len(actual) == len(expected)
        for l,r in zip(actual,expected,strict=True):
            assert l._base is r._base and footprint(l) == footprint(r) and torch.equal(l,r)
            assert l.untyped_storage()._cdata == flat.untyped_storage()._cdata
        results[label] = len(actual)
    # CPU validates the view algorithm above. NPU also tests the compiled guards.
    if device == "npu":
        zero_ref = torch.empty(0, dtype=dtype, device=device)
        for label,flat,shapes in (
            ("empty", makeflat(), [zero_ref] + refs),
            ("float", makeflat().float(), refs),
            ("strided_flat", torch.empty_strided((n,), (2,), dtype=dtype, device=device).zero_(), refs),
        ):
            actual = checked_call(flat, shapes, False, device)
            expected = native_reference(flat, shapes)
            assert len(actual) == len(expected)
            for l,r in zip(actual,expected,strict=True):
                assert footprint(l) == footprint(r) and torch.equal(l,r)
                if l.numel():
                    assert l.untyped_storage()._cdata == flat.untyped_storage()._cdata
                else:
                    assert l.untyped_storage()._cdata != flat.untyped_storage()._cdata
            results[label] = len(actual)
        grad = makeflat().requires_grad_(True)
        expected_grad = grad.detach().clone().requires_grad_(True)
        outs = checked_call(grad, refs, False, device)
        baseline = native_reference(expected_grad, refs)
        sum((o.float().sum() * (i+1) for i,o in enumerate(outs))).backward()
        sum((o.float().sum() * (i+1) for i,o in enumerate(baseline))).backward()
        assert torch.equal(grad.grad, expected_grad.grad)
        results["backward_fallback"] = True
        with torch.autograd.forward_ad.dual_level():
            primal = makeflat()
            tangent = makeflat() + 1
            dual = torch.autograd.forward_ad.make_dual(primal, tangent)
            outs = checked_call(dual, refs, False, device)
            for o, ref in zip(outs,native_reference(tangent,refs),strict=True):
                assert torch.equal(torch.autograd.forward_ad.unpack_dual(o).tangent,ref)
        results["forward_ad_fallback"] = True
        try:
            native(makeflat()[:-1], refs)
        except RuntimeError:
            results["short_flat_fallback_error"] = True
        else:
            raise AssertionError("too-short flat did not raise")
    return {"mode":device, "torch":torch.__version__,
            "unflatten_switch":os.environ.get("NPU_EXPERIMENT_UNFLATTEN_VIEWS"), "tests":results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",choices=("cpu","npu"),required=True)
    print(json.dumps(gate(ap.parse_args().device),indent=2))
