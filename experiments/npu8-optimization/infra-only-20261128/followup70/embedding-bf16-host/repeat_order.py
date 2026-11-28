"""Disambiguate bit-level representation and inter-core FP32 reduction order using fixed buffers."""
import collections
import ctypes as c
import gzip
import hashlib
import json

import torch
import torch_npu
import numpy as np
from pathlib import Path
import sys
from probe_common import P, U, apis, descriptor, here, meta, output, schedule


def sha(tensor):
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


class Trial:
    def __init__(self, shape, table, bank, host, stream, pad=-1):
        self.shape = shape
        self.stream = stream
        self.pad = pad
        self.grad = bank.reshape(*shape, 4, 64).npu()
        self.idx = host.npu()[:, :, table, :]
        self.out = torch.empty((64, 64), dtype=torch.bfloat16, device=self.grad.device)
        self.buf = torch.empty(24 * 1024 * 1024, dtype=torch.uint8, device=self.grad.device)
        self.receipt = dict(shape=shape, table=table, idx_sha256=sha(host[:, :, table, :]),
                            grad_sha256=sha(bank), stream=stream.npu_stream, fixed_grad=self.grad.data_ptr(),
                            fixed_idx=self.idx.data_ptr(), fixed_out=self.out.data_ptr(), fixed_workspace=self.buf.data_ptr())
        self.sizes = {}

    def call(self, leg):
        prepare, launch = apis[leg]
        self.out.fill_(-73)
        torch.npu.synchronize()
        raw = [descriptor(t) for t in (self.grad, self.idx, self.out)]
        size, executor = U(), P()
        try:
            rc = prepare(*raw[:2], 64, self.pad, False, raw[2], c.byref(size), c.byref(executor))
            assert rc == 0 and executor.value, ("prepare", rc, leg)
            assert size.value <= self.buf.numel(), size.value
            self.sizes[leg] = int(size.value)
            rc = launch(self.buf.data_ptr(), size, executor, self.stream.npu_stream)
            assert rc == 0, ("launch", rc, leg)
            torch.npu.synchronize()
        finally:
            for ptr in raw:
                assert meta.aclDestroyTensor(ptr) == 0
        return self.out.cpu().view(torch.int16).clone()


def replay_original():
    # Recreate the original host generator state, including the unused randint draws before randn.
    gen = torch.Generator().manual_seed(1234)
    shape = [2, 22]
    for k, (b, l) in enumerate(schedule + schedule + [shape]):
        host = torch.randint(0, 32, (b, l, 1024, 4), generator=gen)
        grad = torch.randint(-5, 6, (b, l, 4, 64), generator=gen).bfloat16()
        if 60 <= k < 120:
            torch.randn(grad.shape, generator=gen)
    host.remainder_(3)
    pattern = torch.tensor([8388608., 1., -8388608., 1., -1., .00390625, -.00390625]).bfloat16()
    grad = pattern.repeat((grad.numel()+len(pattern)-1)//len(pattern))[:grad.numel()].reshape_as(grad)
    return shape, (17*120+15) % 1024, grad, host


def targeted(shape, cross):
    # Original HSTU Level1 has 40 AIV blocks. Check targeted physical launches in this same probe.
    rows = 4 * shape[0] * shape[1]
    stride = rows // 40
    start = ((39 * stride + 3) // 4) * 4
    sep = ((stride + 3) // 4) * 4
    assert rows - start > 8
    positions = ([0, sep, 2*sep] if cross else [start, start+4, start+8])
    host = torch.ones((*shape, 1024, 4), dtype=torch.int64)
    ids = torch.ones(rows, dtype=torch.int64)
    ids[positions] = 0
    host[:, :, 15, :] = ids.reshape(*shape, 4)
    grad = torch.zeros((rows, 64), dtype=torch.bfloat16)
    # Same finite inputs in each channel. Halfway addition exposes a reorder in FP32.
    for i, val in zip(positions, (2.**24, 1., -2.**24)):
        # The real per-table mean backward broadcasts one token's value to its four features.
        grad[i:i+4] = val
    return 15, grad, host, positions


def repeat(label, trial, points, order, raw):
    results = []
    for leg in order:
        vals = trial.call(leg)
        words = [int(vals[pos]) & 0xffff for pos in points]
        results.append(dict(leg=leg, output_sha256=sha(vals), bf16_words=words))
        raw.write(vals.numpy().tobytes())
    by_leg = {}
    for leg in "AB":
        selected = [x for x in results if x["leg"] == leg]
        hist = [dict(sorted(collections.Counter(x["bf16_words"][i] for x in selected).items())) for i in range(len(points))]
        by_leg[leg] = dict(repetitions=len(selected), distinct_outputs=len({x["output_sha256"] for x in selected}), histograms=hist)
    report["cases"].append(dict(label=label, receipt=trial.receipt, workspace=trial.sizes, points=points,
                                order=order, by_leg=by_leg, results=results))
    output.write_text(json.dumps(report, indent=2))


def rerun_encoding(label, shape, i, sample, stream, raw):
    # Use exactly one nonzero source row for each offending BF16 input pattern.
    words = [4096 * i + 64*p["pos"][0] + p["pos"][1] for p in sample]
    host = torch.zeros((*shape, 1024, 4), dtype=torch.int64)
    grad = torch.zeros((*shape, 4, 64), dtype=torch.bfloat16)
    grad.view(torch.int16).view(-1)[:len(words)] = torch.tensor(words, dtype=torch.int32).to(torch.int16)
    t = Trial(shape, 15, grad, host, stream)
    t.receipt["input_words"] = words
    repeat(label, t, [(0,i) for i in range(len(words))], "ABBABAAB"*2, raw)


def workload_fan_in():
    from collections import Counter
    config = json.loads((Path(sys.argv[1]).parent/"model_many_tables.json").read_bytes())
    gen = torch.Generator().manual_seed(1234)
    per_step = []
    for step, (b, length) in enumerate(schedule):
        torch.randint(1, config["vocab_size"], (b, length+1), generator=gen)
        torch.randint(0, config["num_user_groups"], (b,), generator=gen)
        table_count, feature_count, rows = (config[k] for k in ("num_tables", "features_per_table", "table_rows"))
        data = torch.randint(0, rows, (b,length,table_count,feature_count), generator=gen)
        # A table is provably order-independent under the <=2 nonzero addends rule only if
        # *every* destination row qualifies, independent of its gradient values.
        ids = data.numpy() + rows*np.arange(table_count).reshape(1,1,-1,1)
        counts = np.bincount(ids.ravel(), minlength=table_count*rows).reshape(table_count,rows)
        max_hist = Counter(map(int,counts.max(axis=1)))
        per_step.append(dict(step=step,shape=[b,length],feature_ids_sha256=sha(data),
                             tables_by_max_fan_in=dict(sorted(max_hist.items())),
                             tables_with_max_at_most_two=sum(n for m,n in max_hist.items() if m<=2)))
    return dict(seed=1234,config=config,steps=per_step)


controlled = json.loads((here/"controlled/results.json").read_bytes())
assert len(controlled["checks"]) == 22
streams = [torch.npu.current_stream(), torch.npu.Stream()]
report = dict(soc=torch.npu.get_device_name(0), cases=[], controlled_passed=controlled["passed"], complete=False,
              raw_format="gzip concatenated native little-endian int16 [64,64] tensors, cases and order as in this JSON")
report["workload_fan_in"] = workload_fan_in()
with gzip.open(output.parent/"replays.i16.gz", "wb") as raw:
    if controlled["passed"]:
        for stream_num, stream in enumerate(streams):
            with torch.npu.stream(stream):
                shape, table, grad, host = replay_original()
                t = Trial(shape, table, grad, host, stream)
                repeat("original_cancellation_stream"+str(stream_num), t, [(0, 2), (0, 3), (1, 6)], "ABBABAAB"*6, raw)
                for shape in ([2,22], [1,19], [4,21]):
                    assert shape in schedule
                    for cross in (False, True):
                        table, grad, host, pos = targeted(shape, cross)
                        t = Trial(shape, table, grad, host, stream)
                        t.receipt["nonzero_positions"] = pos
                        repeat(("cross" if cross else "within")+"_block", t, [(0,0)], "ABBABAAB"*4, raw)
    else:
        for row in controlled["checks"]:
            if any(row["diff_AA_BB_AB_AB"]) and row["label"].startswith("all_encodings_") and row["examples"]:
                page = int(row["label"].rsplit("_", 1)[-1])
                for k, shape in enumerate(([1,19],[2,22],[4,21])):
                    with torch.npu.stream(streams[k%2]):
                        rerun_encoding(row["label"], shape, page, row["examples"], streams[k%2], raw)

prof = torch_npu.profiler
with prof.profile(activities=[prof.ProfilerActivity.CPU, prof.ProfilerActivity.NPU], record_shapes=True,
                  experimental_config=prof._ExperimentalConfig(profiler_level=prof.ProfilerLevel.Level1),
                  on_trace_ready=prof.tensorboard_trace_handler(str(output.parent/"profile"))):
    for k, shape in enumerate(([2,22],[1,19],[4,21])):
        table, grad, host, _ = targeted(shape, True)
        stream = streams[k%2]
        with torch.npu.stream(stream):
            t = Trial(shape, table, grad, host, stream)
            for leg in "ABBA":
                with torch.profiler.record_function("EmbeddingFixed_"+leg+"_"+str(shape[0])+"x"+str(shape[1])):
                    t.call(leg)
report["complete"] = True
output.write_text(json.dumps(report, indent=2))

