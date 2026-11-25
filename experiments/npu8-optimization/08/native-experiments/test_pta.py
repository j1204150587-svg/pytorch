"""Validate the two native paths using real card allocations and same-stream consumers."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu


def check_fresh_nd_copy():
    source = torch.arange(360, device="npu", dtype=torch.int32).to(torch.bfloat16)
    first, second, third = source.split((17, 147, 196))
    partitions = [first.view(17, 1), second.view(7, 21), third.view(14, 14)]
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        flat = torch._utils._flatten_dense_tensors(partitions)
        reversed_flat = torch._utils._flatten_dense_tensors(partitions[::-1])
    assert [e.name for e in prof.events()].count("NPUDenseFlattenStorageCopy") == 1
    assert torch.equal(flat, source)
    assert flat.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert torch.equal(reversed_flat, torch.cat((third, second, first)))
    flat.zero_()
    assert int(source[1].item()) == 1
    return {"values": "bitwise", "fresh_storage": True, "nonadjacent_fallback": True}


def check_order_and_lifetime():
    target = torch.zeros(4096, device="npu", dtype=torch.bfloat16)
    graph = torch_npu.npu.NPUGraph()
    with torch_npu.npu.graph(graph):
        output = target * 2
    snapshots = []
    for index in range(1, 33):
        value = torch.empty_like(target)
        value.fill_(index)
        assert graph._try_replay_with_copy(target, value)
        del value
        # Try to reuse the just-released source while graph consumers are pending.
        torch.empty_like(target).fill_(-4)
        snapshots.append(output.clone())
    with torch.npu.stream(torch.npu.Stream()):
        assert not graph._try_replay_with_copy(target, torch.zeros_like(target))
    # No explicit host/device fence before releasing the graph.
    graph.reset()
    for index, output_copy in enumerate(snapshots, 1):
        assert torch.equal(output_copy.cpu(), torch.full((4096,), index * 2, dtype=torch.bfloat16))
    return {"replays": 32, "values": "bitwise", "foreign_stream_rejected": True, "reset_after_enqueue": True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variant", choices=("queued", "flat"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.npu.set_device(0)
    result = check_order_and_lifetime() if args.variant == "queued" else check_fresh_nd_copy()
    args.output.write_text(json.dumps({"torch_npu": torch_npu.__file__, **result}, indent=2) + "\n")


if __name__ == "__main__":
    main()
