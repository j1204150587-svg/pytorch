"""Recompute candidate-vs-legacy scan semantics without DeepSpeed or an NPU."""

import os
os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"

from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch

from overflow import ACL_FORMAT_ND, ScanStats, batched_scan, finish_run


CPU = torch.device("cpu")


def reference_tensor(grad):
    converted = grad.float()
    return converted.isnan().logical_or(converted.isinf()).float().max()


def reference(groups):
    invalid_count = torch.zeros([1], dtype=torch.float, device=CPU)
    for group in groups:
        for grad in group:
            if grad is not None:
                invalid_count += reference_tensor(grad)
    return invalid_count.bool()


def bf16(values):
    return torch.tensor(values, dtype=torch.bfloat16)


class OverflowTest(unittest.TestCase):
    def scan(self, groups, stats=None, **kwargs):
        stats = ScanStats() if stats is None else stats
        return batched_scan(groups, CPU, reference_tensor, lambda: reference(groups), stats,
                            chunk_elements=4, **kwargs)

    def test_finite_and_nonfinite_each_chunk(self):
        for value in (0.0, torch.finfo(torch.bfloat16).max, float("nan"), float("inf"), -float("inf")):
            for group_index in range(3):
                with self.subTest(value=value, group_index=group_index):
                    groups = [[None, bf16([0, 1]), bf16([-1, 0])], [bf16([0, 1, 2, 3])],
                              [bf16([0]), bf16([1, 0, -1])]]
                    groups[group_index][-1][-1] = value
                    stats = ScanStats()
                    original = [grad.clone() for group in groups for grad in group if grad is not None]
                    actual = self.scan(groups, stats)
                    self.assertTrue(torch.equal(actual, reference(groups)))
                    self.assertEqual((actual.shape, actual.dtype, actual.device), ((1,), torch.bool, CPU))
                    self.assertEqual(bool(actual[0]), value in (float("inf"), -float("inf")) or value != value)
                    self.assertEqual((stats.calls, stats.fast_calls, stats.chunks), (1, 1, 3))
                    for grad, saved in zip((grad for group in groups for grad in group if grad is not None),
                                           original, strict=True):
                        torch.testing.assert_close(saved, grad, rtol=0, atol=0, equal_nan=True)

    def test_seeded_mixed_shards(self):
        rng = torch.Generator(device="cpu").manual_seed(1234)
        for trial in range(64):
            groups = [[None, torch.randn((2, 2), generator=rng).to(torch.bfloat16)],
                      [torch.randn((3,), generator=rng).to(torch.bfloat16), bf16(1.0)]]
            if trial % 2:
                groups[1][0][trial % 3] = -float("inf")
            self.assertTrue(torch.equal(self.scan(groups), reference(groups)))

    def test_fallback_preserves_reference(self):
        cases = (
            ("not_bf16", [bf16([1]), torch.tensor([float("inf")])], None),
            ("unsupported_layout_or_view", [bf16([[0, 1], [2, 3]]).t()], None),
            ("unsupported_layout_or_view", [bf16([1]).requires_grad_()], None),
            ("tensor_exceeds_chunk", [bf16([1, 2, 3, 4, 5])], None),
            ("no_gradients", [None], None),
            ("unsupported_npu_storage_format", [bf16([1])], lambda grad: 29),
        )
        for reason, group, storage_format in cases:
            with self.subTest(reason=reason):
                stats = ScanStats()
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    self.assertTrue(torch.equal(self.scan([group], stats, storage_format=storage_format),
                                                reference([group])))
                self.assertEqual((stats.fallback_calls, stats.fast_calls, stats.errors), (1, 0, 0))
                self.assertEqual(stats.reasons, {reason: 1})
                self.assertEqual(json.loads(stderr.getvalue())["reason"], reason)

    def test_dense_npu_metadata_contract(self):
        stats = ScanStats()
        result = self.scan([[None, bf16([1]), bf16([2])]], stats, storage_format=lambda grad: ACL_FORMAT_ND)
        self.assertFalse(result.item())
        self.assertEqual(stats.first_npu_formats, {ACL_FORMAT_ND: 2})
        self.assertIn({"dtype": "torch.bfloat16", "device": "cpu", "layout": "torch.strided", "count": 2},
                      stats.first_shards)

    def test_empty_tensor_preserves_legacy_error(self):
        groups = [[bf16([1]), bf16([])]]
        with self.assertRaises(Exception) as legacy:
            reference(groups)
        stats = ScanStats()
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaises(type(legacy.exception)) as actual:
            self.scan(groups, stats)
        self.assertEqual(str(actual.exception), str(legacy.exception))
        self.assertEqual((stats.errors, stats.fallback_calls, stats.fast_calls), (1, 1, 0))
        fallback, error = (json.loads(line) for line in stderr.getvalue().splitlines())
        self.assertEqual((fallback["reason"], error["path"]), ("empty_tensor", "legacy_fallback"))

    def test_backend_error_is_not_swallowed_or_retried(self):
        def broken(grad):
            raise RuntimeError("simulated backend error")

        def legacy():
            self.fail("backend failure must not be retried after dispatch")

        stats = ScanStats()
        stderr = io.StringIO()
        with redirect_stderr(stderr), self.assertRaisesRegex(RuntimeError, "simulated backend error"):
            batched_scan([[bf16([1]), bf16([2])]], CPU, broken, legacy, stats)
        self.assertEqual((stats.errors, stats.fallback_calls, stats.fast_calls), (1, 0, 0))
        self.assertEqual(json.loads(stderr.getvalue())["path"], "batched")

    def test_measurement_gate_and_failed_run_evidence(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            out = Path(directory)
            valid = ScanStats(calls=2, fast_calls=2, chunks=2)
            finish_run(valid, out, 2, True)
            report = out / "overflow_scan.json"
            self.assertTrue(json.loads(report.read_text())["valid"])
            mixed = ScanStats(calls=2, fast_calls=1, fallback_calls=1, chunks=1)
            with self.assertRaisesRegex(RuntimeError, "invalid candidate run"):
                finish_run(mixed, out, 2, True)
            self.assertEqual(json.loads(report.read_text())["fallback_calls"], 1)
            finish_run(mixed, out, 2, False)
            self.assertFalse(json.loads(report.read_text())["training_completed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
