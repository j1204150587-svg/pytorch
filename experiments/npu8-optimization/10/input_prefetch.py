"""Run the training entry's CPU batch factory up to one step ahead."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


def _prefetched_batches(executor, make_cpu_inputs, steps):
    def materialize():
        return tuple(make_cpu_inputs())

    if not steps:
        return
    pending = executor.submit(materialize)
    for step in range(steps):
        batch = pending.result()
        # The worker can build the next batch while the caller copies and
        # trains on this one. No task is submitted past the final step.
        if step + 1 < steps:
            pending = executor.submit(materialize)
        yield batch


@contextmanager
def cpu_input_batches(make_cpu_inputs, steps, prefetch):
    """Yield ordered batches. Only the consumer should move tensors to a device."""
    if not prefetch:
        # Keep tensors lazy so the caller interleaves CPU generation and copies
        # in the same order as the original inline training loop.
        yield (make_cpu_inputs() for _ in range(steps))
        return

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="hstu-cpu-input")
    batches = _prefetched_batches(executor, make_cpu_inputs, steps)
    try:
        yield batches
    finally:
        batches.close()
        executor.shutdown(wait=True, cancel_futures=True)
