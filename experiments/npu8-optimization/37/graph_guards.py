"""Overlap backward-hook validation with an isolated training step's update."""

from concurrent.futures import ThreadPoolExecutor
from time import perf_counter_ns


class HookLookahead:
    def __init__(self, check, require_fixed_optimizer):
        self.check = check
        self.require_fixed_optimizer = require_fixed_optimizer
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hstu-replay-prep")
        self.worker.submit(lambda: None).result()
        self.pending = None
        self.submitted = self.consumed = self.ready = self.synchronous = 0
        self.worker_ns = self.wait_ns = 0

    def _check(self):
        begin = perf_counter_ns()
        self.check()
        return perf_counter_ns() - begin

    def prepare(self, generation):
        if self.pending is not None:
            raise RuntimeError("unconsumed replay preparation")
        self.require_fixed_optimizer()
        self.pending = generation, self.worker.submit(self._check)
        self.submitted += 1

    def consume(self, generation):
        if self.pending is None:
            self.check()
            self.synchronous += 1
            return
        expected, future = self.pending
        if expected != generation:
            raise RuntimeError("replay preparation does not match the next generation")
        self.require_fixed_optimizer()
        self.ready += future.done()
        begin = perf_counter_ns()
        self.worker_ns += future.result()
        self.wait_ns += perf_counter_ns() - begin
        self.pending = None
        self.consumed += 1

    def finish(self, expected):
        self.worker.shutdown()
        if self.pending is not None or self.submitted != self.consumed or self.consumed + self.synchronous != expected:
            raise RuntimeError("incomplete replay preparation")
        return {key: getattr(self, key) for key in (
            "submitted", "consumed", "ready", "synchronous", "worker_ns", "wait_ns",
        )}
