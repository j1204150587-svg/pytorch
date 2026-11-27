# AOT ID: ['0_inference']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch_npu._C import _npu_getCurrentRawStreamNoWait as get_raw_stream
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch_npu._C import _npu_getCurrentRawStreamNoWait as get_raw_stream
import torch_npu
torch_npu.npu._initialized = torch_npu.npu.is_initialized()
has_initialized = False
import torch_npu._inductor.runtime.triton_heuristics as triton_heuristics
from torch_npu._inductor.experimental.python_wrapper_fast_launch.bind import bind_python_wrapper_kernel_fast

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
assert_alignment = torch._C._dynamo.guards.assert_alignment
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cpu_pinned = torch._C._dynamo.guards._empty_strided_cpu_pinned
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
empty_strided_mtia = torch._C._dynamo.guards._empty_strided_mtia
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p
_npu_fast_launch_metadata_0 = {'graph_id': 'graph', 'callsite_id': 'graph:0', 'callsite_index': 0, 'kernel_name': 'triton_poi_fused_add_mul_0', 'schema_hash': 'fe6b88100a3b5f47', 'runtime_arg_count': 6, 'arg_kinds': (), 'schema_state': 'incomplete', 'schema_reason': 'codegen_signature_missing', 'eligible': True, 'fallback_reason': None}
_npu_fast_launch_call_0 = [None]


# kernel path: /artifacts/dynamic-infra-20261127/pr42634/bench-dynamic-1795771912409431495/00-B/torchinductor_cache_dir/r2/cr2kyz7m3p4tnytdqc6boy3qu5fwovcgyg7nwegnelv4ajj3tn7l.py
# Topologically Sorted Source Nodes: [mul, add], Original ATen: [aten.mul, aten.add]
# Source node to ATen node mapping:
#   add => add_3
#   mul => mul
# Graph fragment:
#   %arg2_1 : Tensor "bf16[1, s27, s53][s27*s53, s53, 1]npu:0" = PlaceHolder[target=arg2_1]
#   %arg3_1 : Tensor "bf16[1, s27, s53][s27*s53, s53, 1]npu:0" = PlaceHolder[target=arg3_1]
#   %mul : Tensor "bf16[1, s27, s53][s27*s53, s53, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%arg2_1, 2.0), kwargs = {})
#   %add_3 : Tensor "bf16[1, s27, s53][s27*s53, s53, 1]npu:0"[num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %arg3_1), kwargs = {})
#   return %add_3
# SchedulerNodes: [SchedulerNode(name='op0')]

triton_poi_fused_add_mul_0 = async_compile.triton('triton_poi_fused_add_mul_0', '''
import triton
import triton.language as tl
from triton.compiler.compiler import AttrsDescriptor

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties

import torch
import torch_npu
if not torch_npu.npu.is_initialized() and torch_npu.npu._is_in_bad_fork():
    torch_npu.npu._initialized = True
from torch_npu._inductor.runtime import triton_heuristics as triton_heuristics
from torch_npu._inductor.runtime import triton_helpers
from torch_npu._inductor.runtime.triton_helpers import libdevice, extension, math as tl_math

@triton_heuristics.pointwise(
    size_hints={'y0': 19, 'x1': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*bf16', 'in_ptr1': '*bf16', 'out_ptr0': '*bf16', 'ks0': 'i64', 'y0_numel': 'i32', 'x1_numel': 'i32', 'Y0BLOCK': 'i32', 'X1BLOCK': 'i32'}, 'device': DeviceProperties(type='npu', index=0, multi_processor_count=40, cc='Ascend910_9362', major=None, regs_per_multiprocessor=None, max_threads_per_multi_processor=None, max_threads_per_block=None, warp_size=None), 'constants': {}, 'mix_mode': 'aiv'},
    inductor_meta={'grid_type': 'GridNpu', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_0', 'mutated_arg_names': [], 'backend_hash': '7E6D1B6CCCB8C8D52D9639951CF7E12BC898581050AFAA3E6320649C48DE8B8A', 'split_axis': [0, 1], 'tiling_axis': [0, 1], 'no_loop_axis': [], 'axis_names': ['y0', 'x1'], 'axis_static_values': (), 'low_dims': {1}, 'numof_reduction_axis': 0, 'split_axis_dtype': torch.bfloat16, 'dual_reduction': False, 'npu_kernel_type': 'simd', 'traced_graph_hash': 'TRACED_GRAPH_HASH', 'traced_graph_dir': 'TRACED_GRAPH_DIR', 'are_deterministic_algorithms_enabled': False, 'inductor_ascend_linear_mode': 'linear', 'runtime_block_arg_names': ('Y0BLOCK', 'X1BLOCK'), 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': False, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'deterministic': False, 'force_filter_reduction_configs': False, 'group_enabled': False, 'group_template': None, 'group_workload': None, 'primary_group_axis': None, 'static_split_axes': (), 'secondary_runtime_symbolic_axes': (), 'group_features': ()},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_mul_0(in_ptr0, in_ptr1, out_ptr0, ks0, y0_numel, x1_numel, Y0BLOCK, X1BLOCK, Y0BLOCK_SUB : tl.constexpr, X1BLOCK_SUB : tl.constexpr):
    y0_offset = tl.program_id(0) * Y0BLOCK
    base_y0= tl.arange(0, Y0BLOCK_SUB)
    loops_y0 = (Y0BLOCK + Y0BLOCK_SUB - 1) // Y0BLOCK_SUB
    x1_offset = tl.program_id(1) * X1BLOCK
    base_x1= tl.arange(0, X1BLOCK_SUB)
    loops_x1 = (X1BLOCK + X1BLOCK_SUB - 1) // X1BLOCK_SUB
    for loop_y0 in range(loops_y0):
        y0 = y0_offset + (loop_y0 * Y0BLOCK_SUB) + base_y0[:,None]
        y0_mask = y0 < min(Y0BLOCK+y0_offset, y0_numel)
        for loop_x1 in range(loops_x1):
            x1 = x1_offset + (loop_x1 * X1BLOCK_SUB) + base_x1[None,:]
            x1_mask = x1 < min(X1BLOCK+x1_offset, x1_numel)
            tmp0 = tl.load(in_ptr0 + (x1 + ks0*y0), x1_mask & y0_mask).to(tl.float32)
            tmp3 = tl.load(in_ptr1 + (x1 + ks0*y0), x1_mask & y0_mask).to(tl.float32)
            tmp1 = 2.0
            tmp2 = tmp0 * tmp1
            tmp4 = tmp2 + tmp3
            tl.store(out_ptr0 + (x1 + ks0*y0), tmp4, x1_mask & y0_mask)
''', device_str='npu')


async_compile.wait(globals())
del async_compile

class Runner:
    def __init__(self, partitions):
        self.partitions = partitions

    def recursively_apply_fns(self, fns):
        new_callables = []
        for fn, c in zip(fns, self.partitions):
            new_callables.append(fn(c))
        self.partitions = new_callables

    def call(self, args):
        arg0_1, arg1_1, arg2_1, arg3_1 = args
        args.clear()
        s27 = arg0_1
        s53 = arg1_1
        with torch.npu.utils.device(0):
            torch.npu.set_device(0)
            buf0 = empty_strided((1, s27, s53), (s27*s53, s53, 1), device='npu', dtype=torch.bfloat16)
            # Topologically Sorted Source Nodes: [mul, add], Original ATen: [aten.mul, aten.add]
            stream0 = get_raw_stream(0)
            _npu_fast_launch_call_0_fn = _npu_fast_launch_call_0[0]
            if _npu_fast_launch_call_0_fn is None:
                _npu_fast_launch_call_0_fn = bind_python_wrapper_kernel_fast(_npu_fast_launch_metadata_0, triton_poi_fused_add_mul_0, call_slot=_npu_fast_launch_call_0)
                _npu_fast_launch_call_0[0] = _npu_fast_launch_call_0_fn
            _npu_fast_launch_call_0_fn(arg2_1, arg3_1, buf0, s53, s27, s53, stream=stream0)
            del arg2_1
            del arg3_1
        return (buf0, )

runner = Runner(partitions=[])
call = runner.call
recursively_apply_fns = runner.recursively_apply_fns


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = 19
    arg1_1 = 64
    arg2_1 = rand_strided((1, 19, 64), (1216, 64, 1), device='npu:0', dtype=torch.bfloat16)
    arg3_1 = rand_strided((1, 19, 64), (1216, 64, 1), device='npu:0', dtype=torch.bfloat16)
    fn = lambda: call([arg0_1, arg1_1, arg2_1, arg3_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
