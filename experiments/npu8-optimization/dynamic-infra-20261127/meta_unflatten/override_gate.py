"""Exercise late third-party registration in isolation, after the sidecar is active."""
import argparse
import json
from pathlib import Path

import torch
import torch_npu
from torch_npu._dynamic_meta_unflatten import counters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    name = "aten::unflatten_dense_tensors"
    if torch._C._dispatch_has_kernel_for_dispatch_key(name, "AutogradMeta"):
        raise ValueError("already has a third-party registration")
    torch.npu.set_device(0)
    flat = torch.arange(80,device="npu:0").to(torch.bfloat16)
    templates = [torch.empty(2,device="meta",dtype=torch.bfloat16) for _ in range(32)]
    probe = torch._utils._unflatten_dense_tensors
    before = counters()
    assert len(probe(flat,templates)) == len(templates)
    active = counters()
    assert active["eligible"] - before["eligible"] == 1

    lib = torch.library.Library("aten","IMPL","AutogradMeta")
    calls = []
    def explicit(buffer, shapes):
        calls.append(len(shapes))
        return [buffer]
    lib.impl("unflatten_dense_tensors",explicit)
    override_result = probe(flat,templates)
    after = counters()
    assert type(override_result) is tuple and len(override_result)==1 and override_result[0] is flat
    assert calls == [len(templates)]
    assert after["fallback"] - active["fallback"] == 1
    args.output.write_text(json.dumps(dict(before=before,active=active,after=after,
                                          late_override_calls=calls),indent=2))


if __name__ == "__main__":
    main()
