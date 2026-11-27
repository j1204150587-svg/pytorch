s17 = 2
s27 = 14
s53 = 64


import os
os.environ['TORCHINDUCTOR_NPU_FAST_LAUNCH'] = '0'
os.environ['TORCHINDUCTOR_CACHE_DIR'] = '/artifacts/dynamic-infra-20261127/pr42634/compiled-cache-guard-off'
os.environ['INDUCTOR_ASCEND_SYMBOLIC_GROUP_AUTOTUNE'] = '0'
os.environ['TORCH_DEVICE_BACKEND_AUTOLOAD'] = '1'
os.environ['TRITON_CACHE_DIR'] = '/artifacts/dynamic-infra-20261127/pr42634/compiled-cache-guard-off/triton/0'
os.environ['TORCHINDUCTOR_NPU_BACKEND'] = 'default'

import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
from math import inf
import torch._inductor.inductor_prims



import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
torch._dynamo.config.assume_static_by_default = False
torch._inductor.config.allow_buffer_reuse = False
torch._inductor.config.post_grad_fusion_options = {'fav3_partition': {}}
torch._inductor.config.deterministic = False
torch._inductor.config.comprehensive_padding = False
torch._inductor.config.triton.coalesce_tiling_analysis = False
torch._inductor.config.triton.store_cubin = False
torch._inductor.config.triton.mix_order_reduction = False
torch._inductor.config.trace.enabled = False
torch._inductor.config.trace.save_real_tensors = False
torch._functorch.config.functionalize_rng_ops = False
torch._functorch.config.fake_tensor_allow_unsafe_data_ptr_access = True
torch._functorch.config.unlift_effect_tokens = True
torch._functorch.config.selective_decompose = False



isolate_fails_code_str = None





if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
# torch version: 2.10.0+cpu
# torch cuda version: None
# torch git version: 449b1768410104d3ed79d3bcfe4ba1d65c7f22c0


# torch.cuda.is_available()==False, no GPU info collected

from torch.nn import *
class Repro(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

    
    
    def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
        mul = torch.ops.aten.mul.Tensor(arg3_1, 2.0);  arg3_1 = None
        add_4 = torch.ops.aten.add.Tensor(mul, arg4_1);  mul = arg4_1 = None
        return (add_4,)
        
def load_args(reader):
    reader.symint(2)  # arg0_1
    reader.symint(14)  # arg1_1
    reader.symint(64)  # arg2_1
    buf0 = reader.storage(None, 2*s27*s53*(s17 - 1) + 2*s53*(s27 - 1) + 2*s53, device=device(type='npu', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf0, (s17, s27, s53), dtype=torch.bfloat16, is_leaf=True)  # arg3_1
    buf1 = reader.storage(None, 2*s27*s53*(s17 - 1) + 2*s53*(s27 - 1) + 2*s53, device=device(type='npu', index=0), dtype_hint=torch.bfloat16)
    reader.tensor(buf1, (s17, s27, s53), dtype=torch.bfloat16, is_leaf=True)  # arg4_1
load_args._version = 0
mod = Repro()
if __name__ == '__main__':
    from torch._dynamo.repro.after_aot import run_repro
    with torch.no_grad():
        run_repro(mod, load_args, accuracy=False, command='run', save_dir=None, tracing_mode='symbolic', check_str=None)
        # To run it separately, do 
        # mod, args = run_repro(mod, load_args, accuracy=False, command='get_args', save_dir=None, tracing_mode='symbolic', check_str=None)
        # mod(*args)