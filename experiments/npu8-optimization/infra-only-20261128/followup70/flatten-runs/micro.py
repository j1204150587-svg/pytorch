"""Compare bit order, output storage, fallback gradients and elapsed windows on one logical NPU."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

out, schedule, recipe_path = map(Path,sys.argv[1:4])
recipe = json.loads(recipe_path.read_bytes())
arm = sys.argv[4]
assert arm in ("A","B")
os.environ.update(recipe["common"])
os.environ[recipe["switch"]] = recipe[arm]
import torch
import torch_npu
library, = {x.split()[-1] for x in Path("/proc/self/maps").read_text().splitlines() if x.endswith("/libtorch_npu.so")}
assert hashlib.sha256(Path(library).read_bytes()).hexdigest()==recipe["libraries"][library]
torch.npu.set_device(0)
flatten = torch._utils._flatten_dense_tensors
stream = torch.npu.Stream()
state = hashlib.sha256()
def check(base, slices):
    expected = torch.cat([t.cpu().reshape(-1) for t in slices]).view(torch.int16)
    got = flatten(slices)
    torch.npu.synchronize()
    assert got.is_contiguous() and torch.equal(got.cpu().view(torch.int16),expected)
    assert got.untyped_storage()._cdata != base.untyped_storage()._cdata
    state.update(expected.numpy().tobytes())
    original = slices[0].cpu().view(torch.int16).clone()
    got.fill_(0)
    assert torch.equal(slices[0].cpu().view(torch.int16),original)

shapes = json.loads(schedule.read_bytes())
for idx,(batch,length) in enumerate(shapes):
    width = batch*length
    bits = ((torch.arange(70*width+5)+idx*997)%65536).short().view(torch.bfloat16)
    base = bits.npu()
    rows = base[3:3+64*width].view(64,width).unbind()
    with torch.npu.stream(stream if idx%2 else torch.npu.current_stream()):
        check(base,list(rows[::-1]))
        check(base,list(rows[1:25])+list(rows[31:63][::-1])+[rows[10],rows[10]])
        check(base,list(rows[::3]))
    # Mixed widths share the same allocation and touch both ends.
    check(base,[base.narrow(0,3+i*width,width).view(batch,length) for i in range(17)]+[base[:2]])

grad = torch.linspace(-1,1,64*16,dtype=torch.float32).npu().requires_grad_()
flat = flatten(list(grad.view(64,16).unbind())[::-1])
flat.sum().backward()
assert torch.equal(grad.grad.cpu(),torch.ones_like(grad.cpu()))
base = torch.arange(256,dtype=torch.float,device="npu").to(torch.bfloat16)
check(base,list(base.view(32,8).unbind()))
check(base,list(base.view(32,8).unbind())[::-1])
check(base,[base[10:17]]*16)
base = torch.zeros(2*4096*512,device="npu",dtype=torch.bfloat16)
rows = list(base.view(1024,64,64).unbind())[::-1]
def window():
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(12):
        flatten(rows)
    torch.npu.synchronize()
    return (time.perf_counter_ns()-start)/12e6
window()
windows = [window() for _ in range(7)]
out.write_text(json.dumps(dict(arm=arm,settings={k:os.environ[k] for k in (recipe["switch"],*recipe["common"])},
    shapes=len(shapes),cases=4*len(shapes),additional_alias_cases=3,data=state.hexdigest(),grad_equal=True,
    library=library,windows_ms=windows,median_ms=statistics.median(windows)),indent=2))
